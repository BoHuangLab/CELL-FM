# -*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

SCRIPT_DIR = Path(__file__).parent
NLS_CSV = "output/hpa/nls_generation/cell_fm_dev/gen_signals.csv"
NES_CSV = "output/hpa/nes_generation/cell_fm_dev/gen_signals.csv"

OUT_DIR  = SCRIPT_DIR / "output"
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


def plot_comparison(nls_freq, nes_freq, nls_counts, nes_counts, out_path: Path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    })

    NLS_COLOR = "#2196F3"
    NES_COLOR = "#F44336"

    nls_tot = sum(nls_counts.values())
    nes_tot = sum(nes_counts.values())
    group_labels = list(GROUPS.keys())

    nls_gpct, nes_gpct, p_vals = [], [], []
    for residues in GROUPS.values():
        nls_grp = sum(nls_counts[aa] for aa in residues)
        nes_grp = sum(nes_counts[aa] for aa in residues)
        nls_gpct.append(nls_grp / nls_tot * 100)
        nes_gpct.append(nes_grp / nes_tot * 100)
        _, p, _, _ = chi2_contingency(np.array([
            [nls_grp, nls_tot - nls_grp],
            [nes_grp, nes_tot - nes_grp],
        ]))
        p_vals.append(p)

    # ── dumbbell plot: groups on Y, frequency on X ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    gy = np.arange(len(GROUPS))

    for i, (nls_v, nes_v, p, label) in enumerate(
            zip(nls_gpct, nes_gpct, p_vals, group_labels)):
        # connecting line
        ax.plot([nls_v, nes_v], [i, i], color="#cccccc", linewidth=3.5, zorder=1)
        # dots
        ax.scatter(nls_v, i, color=NLS_COLOR, s=220, zorder=3, label="NLS" if i == 0 else "")
        ax.scatter(nes_v, i, color=NES_COLOR, s=220, zorder=3, label="NES" if i == 0 else "")
        # value labels
        if nls_v > nes_v:
            ax.text(nls_v + 0.1, i + 0.18, f"{nls_v:.1f}%", ha="center", va="bottom",
                    fontsize=18, color=NLS_COLOR, fontweight="bold")
            ax.text(nes_v - 0.1, i - 0.18, f"{nes_v:.1f}%", ha="center", va="top",
                    fontsize=18, color=NES_COLOR, fontweight="bold")
        else:
            ax.text(nls_v - 0.1, i + 0.18, f"{nls_v:.1f}%", ha="center", va="bottom",
                    fontsize=18, color=NLS_COLOR, fontweight="bold")
            ax.text(nes_v + 0.1, i + 0.18, f"{nes_v:.1f}%", ha="center", va="bottom",
                    fontsize=18, color=NES_COLOR, fontweight="bold")

    ax.set_yticks(gy)
    ax.set_yticklabels(group_labels, rotation=45, ha="right")
    ax.set_xlabel("Frequency (%)")
    ax.set_title("Residue group comparison: NLS vs. NES")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=4, width=1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    x_max = max(max(nls_gpct), max(nes_gpct))
    ax.set_xlim(0, x_max * 1.55)
    ax.set_ylim(-0.6, len(GROUPS) - 0.4)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


GROUPS = {
    "(K+R)":       list("KR"),
    "(L+I+V+F+M)": list("LIVFM"),
}


def group_stats(nls_freq: dict[str, float], nes_freq: dict[str, float],
                nls_counts: dict[str, int], nes_counts: dict[str, int]):
    print("\n── Group statistics ─────────────────────────────────────────────")
    print(f"{'Group':<26}  {'NLS (%)':>9}  {'NES (%)':>9}  {'Δ (pp)':>9}  p-value")
    print("-" * 72)
    for label, residues in GROUPS.items():
        nls_pct = sum(nls_freq[aa] for aa in residues) * 100
        nes_pct = sum(nes_freq[aa] for aa in residues) * 100
        # 2×2 chi-squared: group vs rest, NLS vs NES
        nls_grp = sum(nls_counts[aa] for aa in residues)
        nes_grp = sum(nes_counts[aa] for aa in residues)
        nls_tot = sum(nls_counts.values())
        nes_tot = sum(nes_counts.values())
        table = np.array([
            [nls_grp,       nls_tot - nls_grp],
            [nes_grp,       nes_tot - nes_grp],
        ])
        _, p, _, _ = chi2_contingency(table)
        print(f"{label:<26}  {nls_pct:>9.3f}  {nes_pct:>9.3f}  {nls_pct - nes_pct:>+9.3f}  {p:.2e}")


def chi2_test(nls_counts: dict[str, int], nes_counts: dict[str, int]):
    table = np.array([
        [nls_counts[aa] for aa in AMINO_ACIDS],
        [nes_counts[aa] for aa in AMINO_ACIDS],
    ])
    chi2, p, dof, _ = chi2_contingency(table)
    print(f"\nChi-squared test (overall composition)")
    print(f"  chi2 = {chi2:.2f},  dof = {dof},  p = {p:.2e}")
    return chi2, p


def print_table(nls_freq, nes_freq):
    header = f"{'AA':>3}  {'NLS (%)':>9}  {'NES (%)':>9}  {'Δ (pp)':>9}"
    print("\n" + header)
    print("-" * len(header))
    for aa in AMINO_ACIDS:
        nls_p = nls_freq[aa] * 100
        nes_p = nes_freq[aa] * 100
        print(f"{aa:>3}  {nls_p:>9.3f}  {nes_p:>9.3f}  {nls_p - nes_p:>+9.3f}")


def plot_aa_diff(nls_freq: dict[str, float], nes_freq: dict[str, float],
                 nls_counts: dict[str, int], nes_counts: dict[str, int],
                 out_path: Path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    nls_tot = sum(nls_counts.values())
    nes_tot = sum(nes_counts.values())

    # sort by absolute difference descending
    diffs = {aa: (nls_freq[aa] - nes_freq[aa]) * 100 for aa in AMINO_ACIDS}
    aas_sorted = sorted(AMINO_ACIDS, key=lambda aa: abs(diffs[aa]), reverse=True)
    diff_vals = [diffs[aa] for aa in aas_sorted]

    # per-AA chi-squared p-values
    p_vals = []
    for aa in aas_sorted:
        nls_c = nls_counts[aa]
        nes_c = nes_counts[aa]
        _, p, _, _ = chi2_contingency(np.array([
            [nls_c, nls_tot - nls_c],
            [nes_c, nes_tot - nes_c],
        ]))
        p_vals.append(p)

    colors = ["#2196F3" if d > 0 else "#F44336" for d in diff_vals]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    x = np.arange(len(aas_sorted))
    bars = ax.bar(x, diff_vals, color=colors, width=0.65, zorder=2)

    # # significance markers above each bar
    # for i, (d, p) in enumerate(zip(diff_vals, p_vals)):
    #     label = _pval_label(p)
    #     if label != "ns":
    #         y_pos = d + (0.05 if d >= 0 else -0.05)
    #         va = "bottom" if d >= 0 else "top"
    #         ax.text(i, y_pos, label, ha="center", va=va, fontsize=13, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(aas_sorted)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Δ frequency (pp)\nNLS − NES")
    ax.set_title("Per-residue frequency difference: NLS vs. NES")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)
    ax.tick_params(axis="x", length=0)

    # legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="NLS-enriched"),
        Patch(color="#F44336", label="NES-enriched"),
    ], frameon=False, fontsize=18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def main():
    nls_seqs = read_sequences(NLS_CSV)
    nes_seqs = read_sequences(NES_CSV)

    print(f"NLS sequences: {len(nls_seqs)},  total residues: {sum(len(s) for s in nls_seqs)}")
    print(f"NES sequences: {len(nes_seqs)},  total residues: {sum(len(s) for s in nes_seqs)}")

    nls_freq   = aa_frequencies(nls_seqs)
    nes_freq   = aa_frequencies(nes_seqs)
    nls_counts = aa_counts(nls_seqs)
    nes_counts = aa_counts(nes_seqs)

    print_table(nls_freq, nes_freq)
    chi2_test(nls_counts, nes_counts)
    group_stats(nls_freq, nes_freq, nls_counts, nes_counts)

    plot_comparison(nls_freq, nes_freq, nls_counts, nes_counts,
                    OUT_DIR / "aa_freq_nls_vs_nes.png")
    plot_aa_diff(nls_freq, nes_freq, nls_counts, nes_counts,
                 OUT_DIR / "aa_freq_diff_nls_vs_nes.png")


if __name__ == "__main__":
    main()
