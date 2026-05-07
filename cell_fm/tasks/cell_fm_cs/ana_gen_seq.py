# llps_phase_sep_analysis.py
# -*- coding: utf-8 -*-
"""
Analyze N groups of generated protein sequences (one sequence per line),
compute LLPS-related features, and produce dataset-level comparisons with plots.

Usage:
  python llps_phase_sep_analysis.py \
    --inputs runA/gen_signals.csv runB/gen_signals.csv \
    --labels runA runB \
    --outdir results_llps

Notes:
- Input is robust: if a line contains multiple sequences separated by spaces,
  the parser will split on whitespace and treat each as an independent sequence.
- Plots use matplotlib only.
"""

import argparse
import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# -------------------- Amino-acid constants & property tables --------------------
AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte–Doolittle hydropathy
HYDRO = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9,
    'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3,
    'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5,
    'K': -3.9, 'R': -4.5
}

# Approx pKa (side chains + termini) for net-charge / pI estimates
PKA = {
    "Cterm": 3.55, "Nterm": 7.50,
    "C": 8.5, "D": 3.9, "E": 4.1, "Y": 10.1, "H": 6.5, "K": 10.5, "R": 12.0
}

# Sets for quick fractions
POLAR       = set("STNQYH")
POSITIVE    = set("KR")
NEGATIVE    = set("DE")
AROMATIC    = set("FWY")
ALIPHATIC   = set("AVLI")
CHARGED     = POSITIVE | NEGATIVE
LC_SET      = set("QNGSP")  # low-complexity proxy (polar/coil-prone)


# -------------------- Parsers & utilities --------------------
def clean_seq(s: str) -> str:
    """Keep only canonical AAs and uppercase."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[^A-Za-z]", " ", s).upper()
    s = "".join([a for a in s if a in AA])
    return s

def read_sequences_from_csv(path: str) -> List[str]:
    """Robust reader: each line may contain one sequence OR multiple sequences split by whitespace."""
    seqs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = re.split(r"\s+", line.strip())
            for tok in tokens:
                seq = clean_seq(tok)
                if seq:
                    seqs.append(seq)
    return seqs


# -------------------- Feature functions --------------------
def aa_counts(seq: str) -> Dict[str, int]:
    c = Counter(seq)
    return {a: c.get(a, 0) for a in sorted(AA)}

def gravy(seq: str) -> float:
    if not seq: return float("nan")
    vals = [HYDRO[a] for a in seq if a in HYDRO]
    return float(np.mean(vals)) if vals else float("nan")

def frac_set(seq: str, S: set) -> float:
    return (sum(a in S for a in seq) / len(seq)) if seq else float("nan")

def net_charge_at_pH(seq: str, pH: float = 7.0) -> float:
    c = Counter(seq)
    pos = 1/(1+10**(pH-PKA["Nterm"])) \
        + c["K"]*(1/(1+10**(pH-PKA['K']))) \
        + c["R"]*(1/(1+10**(pH-PKA['R']))) \
        + c["H"]*(1/(1+10**(pH-PKA['H'])))
    neg = 1/(1+10**(PKA["Cterm"]-pH)) \
        + c["D"]*(1/(1+10**(PKA['D']))) \
        + c["E"]*(1/(1+10**(PKA['E']))) \
        + c["C"]*(1/(1+10**(PKA['C']))) \
        + c["Y"]*(1/(1+10**(PKA['Y'])))
    return float(pos - neg)

def isoelectric_point(seq: str, lo: float = 2.0, hi: float = 12.0, steps: int = 30) -> float:
    if not seq: return float("nan")
    for _ in range(steps):
        mid = 0.5*(lo+hi)
        ch  = net_charge_at_pH(seq, mid)
        if ch > 0: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def fcr(seq: str) -> float:
    """Fraction of charged residues."""
    if not seq: return float("nan")
    return (sum(a in CHARGED for a in seq)) / len(seq)

def ncpr(seq: str) -> float:
    """Net charge per residue = (K+R-D-E)/L (H excluded by convention; include if desired)."""
    if not seq: return float("nan")
    c = Counter(seq)
    pos = c['K'] + c['R']
    neg = c['D'] + c['E']
    return (pos - neg) / len(seq)

def scd(seq):
    """
    Sequence Charge Decoration (SCD) as in Sawle & Ghosh:
    SCD = (1/N) * sum_{m=2..N} sum_{n=1..m-1} q_m q_n (m-n)^{1/2}
    where q_i = +1 for K,R,H and -1 for D,E, 0 otherwise.
    notice: in this definition,
    - more blocky (segregated) -> more negative SCD (smaller value)
    - more mixed (alternating)  -> SCD closer to 0 or slightly positive    
    """
    charge_map = {'K': 1, 'R': 1, 'D': -1, 'E': -1}
    q = np.array([charge_map.get(a, 0) for a in seq])
    N = len(q)

    SCD = 0.0
    for m in range(1, N):          # m = 2..N (0-based index)
        for n in range(0, m):      # n = 1..m-1
            SCD += q[m] * q[n] * np.sqrt(m - n)

    return SCD / N

def motif_density(seq: str, motif: str) -> float:
    """Count overlapping motif occurrences per residue."""
    if not seq or not motif: return float("nan")
    m = 0
    k = len(motif)
    for i in range(0, len(seq) - k + 1):
        if seq[i:i+k] == motif: m += 1
    return m / len(seq)

def any_kmer_density(seq: str, kmers: List[str]) -> float:
    """Sum densities of multiple motifs (overlapping allowed)."""
    return float(np.nansum([motif_density(seq, m) for m in kmers]))


# -------------------- Per-sequence feature extraction --------------------
def features_for_sequence(seq: str) -> Dict[str, float]:
    seq = clean_seq(seq)
    comp = aa_counts(seq)
    return {
        "length": len(seq),
        "gravy": gravy(seq),
        "frac_polar": frac_set(seq, POLAR),
        "frac_charged": fcr(seq),
        "frac_positive": frac_set(seq, POSITIVE),
        "frac_negative": frac_set(seq, NEGATIVE),
        "frac_aromatic": frac_set(seq, AROMATIC),
        "lc_fraction": frac_set(seq, LC_SET),          # low-complexity proxy
        "net_charge_pH7": net_charge_at_pH(seq, 7.0),
        "pI": isoelectric_point(seq),
        "NCPR": ncpr(seq),
        "SCD": scd(seq),
        # Sticker proxies:
        "frac_R": seq.count("R")/len(seq) if len(seq) else np.nan,
        "frac_K": seq.count("K")/len(seq) if len(seq) else np.nan,
        "frac_Y": seq.count("Y")/len(seq) if len(seq) else np.nan,
        "frac_F": seq.count("F")/len(seq) if len(seq) else np.nan,
        "frac_W": seq.count("W")/len(seq) if len(seq) else np.nan,
        "RG_density": motif_density(seq, "RG"),
        "RGG_density": motif_density(seq, "RGG"),
        "R_aromatic_mix": (seq.count("R") * (seq.count("Y")+seq.count("F")+seq.count("W"))) / (len(seq)**2) if len(seq) else np.nan,
        **comp,
    }


# -------------------- Plot helpers --------------------
def violin_plot(df: pd.DataFrame, x: str, y: str, outpath: str, title: str):
    groups = sorted(df[x].unique())
    data = [df[df[x]==g][y].dropna().values for g in groups]

    plt.figure(figsize=(7,5))
    plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks(np.arange(1, len(groups)+1), groups, rotation=20)
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def bar_means_with_sem(df: pd.DataFrame, group_col: str, metric: str, outpath: str, title: str):
    agg = df.groupby(group_col)[metric].agg(['mean','count','std'])
    agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
    xs = np.arange(len(agg))
    plt.figure(figsize=(7,5))
    plt.bar(xs, agg['mean'].values, yerr=agg['sem'].values)
    plt.xticks(xs, agg.index, rotation=20)
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel(group_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def scatter(df: pd.DataFrame, x: str, y: str, hue: str, outpath: str, title: str):
    plt.figure(figsize=(6,5))
    groups = sorted(df[hue].unique())
    for g in groups:
        sub = df[df[hue]==g]
        plt.scatter(sub[x], sub[y], label=str(g), alpha=0.7)
    plt.xlabel(x); plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -------------------- Main --------------------
def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("If --labels is provided, it must match --inputs length.")

    labels = args.labels if args.labels else [f"group_{i+1:02d}" for i in range(len(args.inputs))]

    # Load all sequences & compute per-seq features
    records = []
    for path, label in zip(args.inputs, labels):
        seqs = read_sequences_from_csv(path)
        for idx, s in enumerate(seqs):
            f = features_for_sequence(s)
            f["group"] = label
            f["seq_id"] = f"{label}_{idx:04d}"
            records.append(f)

    per_seq = pd.DataFrame(records)
    per_seq.to_csv(os.path.join(args.outdir, "per_sequence_features.csv"), index=False)

    # Group summary
    numeric_cols = per_seq.select_dtypes(include=[np.number]).columns.tolist()

    def q25(x): return x.quantile(0.25)
    q25.__name__ = "q25"
    def q75(x): return x.quantile(0.75)
    q75.__name__ = "q75"

    group_summary = (
        per_seq
        .groupby("group")[numeric_cols]
        .agg(["mean", "std", "median", q25, q75])  # 传“函数列表”
    )

    group_summary.columns = [
        f"{col}_{stat}" for col, stat in group_summary.columns.to_flat_index()
    ]

    group_summary.to_csv(os.path.join(args.outdir, "group_summary.csv"))


    # Plots (per metric)
    metrics = [
        "GRAVY_placeholder",  # to keep order; real key is "gravy"
        "gravy", "pI", "net_charge_pH7",
        "frac_charged", "NCPR", "SCD",
        "frac_R", "frac_K", "frac_Y", "frac_F", "frac_W",
        "RG_density", "RGG_density",
        "lc_fraction", "frac_polar", "frac_aromatic", "length",
    ]
    metrics = [m for m in metrics if m in per_seq.columns]

    # Violin plots + bar means
    for m in metrics:
        violin_plot(per_seq, "group", m, os.path.join(args.outdir, f"violin_{m}.png"), f"Violin: {m} by group")
        bar_means_with_sem(per_seq, "group", m, os.path.join(args.outdir, f"bar_means_{m}.png"), f"Group means (±SEM): {m}")

    # Key scatters
    if set(["NCPR","frac_charged"]).issubset(per_seq.columns):
        scatter(per_seq, "NCPR", "frac_charged", "group",
                os.path.join(args.outdir, "scatter_NCPR_vs_FCR.png"),
                "NCPR vs FCR")
    if set(["frac_R","frac_Y","frac_F","frac_W"]).issubset(per_seq.columns):
        per_seq["Aromatic_total"] = per_seq["frac_Y"].fillna(0)+per_seq["frac_F"].fillna(0)+per_seq["frac_W"].fillna(0)
        scatter(per_seq, "frac_R", "Aromatic_total", "group",
                os.path.join(args.outdir, "scatter_Arg_vs_Aromatic.png"),
                "Arg fraction vs Aromatic fraction")

    print(f"[INFO] Sequences analyzed: {len(per_seq)} from {len(args.inputs)} groups.")
    print(f"[INFO] Outputs written to: {args.outdir}")
    print(f"[INFO] Columns available: {list(per_seq.columns)}")
    print("[INFO] Key metrics plotted:", ", ".join(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=None, help="Paths to gen_signals.csv files (one sequence per line; whitespace inside a line is allowed).")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional group labels matching --inputs. If omitted, will be group_1, group_2, ...")
    parser.add_argument("--outdir", type=str, default="output/condenseq/img2seq/results_llps", help="Output directory for CSVs and plots.")
    args = parser.parse_args()

    args.inputs = sorted(glob('output/condenseq/img2seq/el_*/gen_signals.csv'))
    print(args.inputs)

    main(args)