# -*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

SCRIPT_DIR   = Path(__file__).parent
NLS_CSV      = "output/hpa/nls_generation/cell_fm_dev/gen_signals.csv"
NES_CSV      = "output/hpa/nes_generation/cell_fm_dev/gen_signals.csv"
BASELINE_CSV = "/hpc/reference/opencell/human_protein_atlas/splits/all_merged_meta_data.csv"

OUT_DIR = SCRIPT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def read_sequences(path: Path) -> list[str]:
    seqs = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                seqs.append(s.upper())
    return seqs


def read_sequences_from_csv(path: str, col: str = "sequence") -> list[str]:
    df = pd.read_csv(path)
    return [s.upper() for s in df[col].dropna() if isinstance(s, str) and s.strip()]


def aa_frequencies(seqs: list[str]) -> dict[str, float]:
    counter: Counter = Counter()
    for seq in seqs:
        counter.update(c for c in seq if c in AMINO_ACIDS)
    total = sum(counter.values())
    return {aa: counter.get(aa, 0) / total for aa in AMINO_ACIDS}


def aa_counts(seqs: list[str]) -> dict[str, int]:
    counter: Counter = Counter()
    for seq in seqs:
        counter.update(c for c in seq if c in AMINO_ACIDS)
    return {aa: counter.get(aa, 0) for aa in AMINO_ACIDS}


def _pval_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def print_table(sig_freq, base_freq, sig_label: str):
    header = (f"{'AA':>3}  {'Base (%)':>9}  {sig_label+' (%)':>12}  {'Δ (pp)':>9}")
    print(f"\n── {sig_label} vs. baseline ──")
    print(header)
    print("-" * len(header))
    for aa in AMINO_ACIDS:
        b_p = base_freq[aa] * 100
        s_p = sig_freq[aa] * 100
        print(f"{aa:>3}  {b_p:>9.3f}  {s_p:>12.3f}  {s_p - b_p:>+9.3f}")


def plot_vs_baseline(
    sig_freq:   dict[str, float],
    base_freq:  dict[str, float],
    sig_counts: dict[str, int],
    base_counts: dict[str, int],
    label:     str,
    color:     str,
    out_path:  Path,
):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    sig_tot  = sum(sig_counts.values())
    base_tot = sum(base_counts.values())

    shifts = {aa: (sig_freq[aa] - base_freq[aa]) * 100 for aa in AMINO_ACIDS}
    aas_sorted = sorted(AMINO_ACIDS, key=lambda aa: abs(shifts[aa]), reverse=True)
    shift_vals = [shifts[aa] for aa in aas_sorted]

    # per-AA chi-squared: signal vs baseline
    p_vals = []
    for aa in aas_sorted:
        _, p, _, _ = chi2_contingency(np.array([
            [sig_counts[aa],  sig_tot  - sig_counts[aa]],
            [base_counts[aa], base_tot - base_counts[aa]],
        ]))
        p_vals.append(p)

    bar_colors = [color if v > 0 else "#aaaaaa" for v in shift_vals]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    x = np.arange(len(aas_sorted))
    ax.bar(x, shift_vals, color=bar_colors, width=0.65, zorder=2)

    # for i, (v, p) in enumerate(zip(shift_vals, p_vals)):
    #     lbl = _pval_label(p)
    #     if lbl != "ns":
    #         y_pos = v + (0.05 if v >= 0 else -0.05)
    #         va = "bottom" if v >= 0 else "top"
    #         ax.text(i, y_pos, lbl, ha="center", va=va, fontsize=13, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(aas_sorted)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Δ frequency (pp)\nvs. proteome baseline")
    ax.set_title(f"Per-residue frequency difference: {label} vs. proteome baseline")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)
    ax.tick_params(axis="x", length=0)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=color,     label=f"{label}-enriched"),
        Patch(color="#aaaaaa", label="depleted"),
    ], frameon=False, fontsize=18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def main():
    nls_seqs  = read_sequences(NLS_CSV)
    nes_seqs  = read_sequences(NES_CSV)
    base_seqs = read_sequences_from_csv(BASELINE_CSV)

    print(f"NLS sequences:     {len(nls_seqs)},  total residues: {sum(len(s) for s in nls_seqs)}")
    print(f"NES sequences:     {len(nes_seqs)},  total residues: {sum(len(s) for s in nes_seqs)}")
    print(f"Baseline proteins: {len(base_seqs)},  total residues: {sum(len(s) for s in base_seqs)}")

    nls_freq    = aa_frequencies(nls_seqs)
    nes_freq    = aa_frequencies(nes_seqs)
    base_freq   = aa_frequencies(base_seqs)
    nls_counts  = aa_counts(nls_seqs)
    nes_counts  = aa_counts(nes_seqs)
    base_counts = aa_counts(base_seqs)

    print_table(nls_freq, base_freq, "NLS")
    print_table(nes_freq, base_freq, "NES")

    plot_vs_baseline(nls_freq, base_freq, nls_counts, base_counts,
                     label="NLS", color="#2196F3",
                     out_path=OUT_DIR / "nls_vs_baseline.png")
    plot_vs_baseline(nes_freq, base_freq, nes_counts, base_counts,
                     label="NES", color="#F44336",
                     out_path=OUT_DIR / "nes_vs_baseline.png")


if __name__ == "__main__":
    main()
