import numpy as np
import random
import pandas as pd

# =========================
# 1. compute kappa
# =========================

import numpy as np

def compute_sigma(f_pos, f_neg):
    """sigma = (f+ - f-)^2 / (f+ + f-)"""
    if f_pos + f_neg == 0:
        return 0.0
    return ((f_pos - f_neg)**2) / (f_pos + f_neg)

def compute_blob_sigmas(seq, g):
    """
    Compute sigma_i for all overlapping blobs of length g
    """
    pos_set = {"K", "R"}
    neg_set = {"D", "E"}

    sigmas = []
    L = len(seq)

    for start in range(L - g + 1):
        fragment = seq[start:start+g]
        f_pos = sum(aa in pos_set for aa in fragment) / g
        f_neg = sum(aa in neg_set for aa in fragment) / g
        sigma_i = compute_sigma(f_pos, f_neg)
        sigmas.append(sigma_i)

    return np.array(sigmas)


def compute_delta(sigmas, sigma_global):
    """
    delta = (1/N) Σ_i (sigma_i - sigma_global)^2
    """
    return np.mean((sigmas - sigma_global)**2)


def compute_kappa_for_g(seq, g):
    """
    Compute kappa for a given blob size g (unnormalized)
    """
    # Find global positive and negative proportions
    pos_set = {"K", "R"}
    neg_set = {"D", "E"}

    L = len(seq)
    n_pos = sum(aa in pos_set for aa in seq)
    n_neg = sum(aa in neg_set for aa in seq)

    f_pos = n_pos / L
    f_neg = n_neg / L
    sigma_global = compute_sigma(f_pos, f_neg)

    # Current sequence's blob sigma_i
    sigmas = compute_blob_sigmas(seq, g)
    delta = compute_delta(sigmas, sigma_global)

    # ------- Compute delta_max: maximum possible delta with the same composition -------
    # According to the paper definition: most blocky (all positives together, all negatives together)
    charged_positions = [aa for aa in seq if aa in pos_set or aa in neg_set]
    # Arrange all positives first, then all negatives
    positives = [aa for aa in charged_positions if aa in pos_set]
    negatives = [aa for aa in charged_positions if aa in neg_set]
    blocky_order = positives + negatives

    # Fill the blocky sequence back into the original charged positions
    seq_blocky = list(seq)
    idx = 0
    for i, aa in enumerate(seq):
        if aa in pos_set or aa in neg_set:
            seq_blocky[i] = blocky_order[idx]
            idx += 1
    seq_blocky = "".join(seq_blocky)

    # blocky sequence's blob sigma_i
    sigmas_blocky = compute_blob_sigmas(seq_blocky, g)
    delta_max = compute_delta(sigmas_blocky, sigma_global)

    if delta_max == 0:
        return 0.0

    return delta / delta_max


def compute_kappa(seq):
    """
    Das-Pappu (2013) original kappa definition:
    kappa = (kappa_g=5 + kappa_g=6) / 2
    """
    k5 = compute_kappa_for_g(seq, 5)
    k6 = compute_kappa_for_g(seq, 6)
    return (k5 + k6) / 2

# =========================
# 2. generated mutant patterns
# =========================

CHARGED_SET = set("KRDE")
POS_SET = set("KR")
NEG_SET = set("DE")

def get_charged_indices(seq):
    return [i for i, a in enumerate(seq) if a in CHARGED_SET]

def extract_charged(seq):
    return [a for a in seq if a in CHARGED_SET]

def make_mutant_from_pattern(seq, charged_res_list):
    """
    put charged_res_list back into seq, keeping non-charged positions unchanged.
    """
    seq_list = list(seq)
    charged_indices = get_charged_indices(seq)
    assert len(charged_indices) == len(charged_res_list)
    for idx, aa in zip(charged_indices, charged_res_list):
        seq_list[idx] = aa
    return "".join(seq_list)

def make_blocky_pattern(seq):
    """
    high kappa: postive charges first, negative charges later
    """
    charged = extract_charged(seq)
    positives = [a for a in charged if a in POS_SET]
    negatives = [a for a in charged if a in NEG_SET]
    # postive charges first, negative charges later
    return positives + negatives

def make_alternating_pattern(seq):
    """
    low kappa: alternate + and - as much as possible at charged positions
    """
    charged = extract_charged(seq)
    positives = [a for a in charged if a in POS_SET]
    negatives = [a for a in charged if a in NEG_SET]
    random.shuffle(positives)
    random.shuffle(negatives)
    pos_i = neg_i = 0

    # Decide whether to start with positive or negative: start with the one that has more
    use_pos_next = len(positives) >= len(negatives)

    new_charged = []
    for _ in charged:
        if use_pos_next:
            if pos_i < len(positives):
                new_charged.append(positives[pos_i])
                pos_i += 1
            elif neg_i < len(negatives):
                new_charged.append(negatives[neg_i])
                neg_i += 1
            use_pos_next = False
        else:
            if neg_i < len(negatives):
                new_charged.append(negatives[neg_i])
                neg_i += 1
            elif pos_i < len(positives):
                new_charged.append(positives[pos_i])
                pos_i += 1
            use_pos_next = True

    return new_charged

def make_random_pattern(seq, rng):
    """
    medium kappa: randomly shuffle all charged residues
    """
    charged = extract_charged(seq)
    rng.shuffle(charged)
    return charged


# =========================
# 3. main function: generate panel
# =========================

def generate_kappa_panel(seq,
                         n_random=2000,
                         panel_size=10,
                         seed=0):
    """
    Inputs:
        seq         : original sequence (string)
        n_random    : how many random scramble mutants to generate
        panel_size  : how many sequences to output in the final panel
        seed        : random seed
    Outputs:
        panel_df: DataFrame containing mutant_id, seq, kappa, type
    """
    rng = random.Random(seed)
    charged_indices = get_charged_indices(seq)
    n_charged = len(charged_indices)
    print(f"Sequence length = {len(seq)}, charged residues = {n_charged}")

    records = []

    # 1) WT
    k_wt = compute_kappa(seq)
    records.append({
        "mutant_id": "WT",
        "seq": seq,
        "kappa": k_wt,
        "type": "WT"
    })

    # 2) blocky (high κ)
    blocky_pattern = make_blocky_pattern(seq)
    seq_blocky = make_mutant_from_pattern(seq, blocky_pattern)
    k_blocky = compute_kappa(seq_blocky)
    records.append({
        "mutant_id": "blocky",
        "seq": seq_blocky,
        "kappa": k_blocky,
        "type": "designed_high_kappa"
    })

    # 3) alternating (low κ)
    alt_pattern = make_alternating_pattern(seq)
    seq_alt = make_mutant_from_pattern(seq, alt_pattern)
    k_alt = compute_kappa(seq_alt)
    records.append({
        "mutant_id": "alternating",
        "seq": seq_alt,
        "kappa": k_alt,
        "type": "designed_low_kappa"
    })

    # 4) multiple random scrambles
    seen_seqs = {seq, seq_blocky, seq_alt}
    for i in range(n_random):
        pattern = make_random_pattern(seq, rng)
        seq_rand = make_mutant_from_pattern(seq, pattern)
        if seq_rand in seen_seqs:
            continue
        seen_seqs.add(seq_rand)
        k = compute_kappa(seq_rand)
        records.append({
            "mutant_id": f"rand_{i}",
            "seq": seq_rand,
            "kappa": k,
            "type": "random"
        })

    df = pd.DataFrame(records)
    # Sort by kappa
    df = df.sort_values("kappa").reset_index(drop=True)

    # 5) Uniformly sample panel_size sequences from min to max kappa
    if panel_size > len(df):
        panel_size = len(df)
    indices = np.linspace(0, len(df) - 1, panel_size).round().astype(int)
    panel_df = df.iloc[indices].reset_index(drop=True)
    panel_df["panel_rank"] = np.arange(1, len(panel_df) + 1)

    return df, panel_df


# =========================
# 4. Example usage
# =========================

if __name__ == "__main__":
    seq = "EDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFG"

    all_df, panel_df = generate_kappa_panel(
        seq,
        n_random=3000,   # You can increase/decrease this
        panel_size=10,   # Desired panel size
        seed=42
    )

    print("\n=== All mutants summary (first 10) ===")
    print(all_df.head(10))

    print("\n=== Kappa panel (low → high) ===")
    print(panel_df[["panel_rank", "mutant_id", "kappa", "type"]])

    # Save to CSV for easy use in experimental design
    all_df.to_csv("all_mutants_kappa.csv", index=False)
    panel_df.to_csv("kappa_panel.csv", index=False)
