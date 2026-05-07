import numpy as np
import random
import pandas as pd

# =========================
# 1. compute SCD (Sawle & Ghosh definition)
# =========================

def compute_scd(seq):
    """
    Sequence Charge Decoration (SCD) as in Sawle & Ghosh:
    SCD = (1/N) * sum_{m=2..N} sum_{n=1..m-1} q_m q_n (m-n)^{1/2}
    where q_i = +1 for K,R and -1 for D,E, 0 otherwise.
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
    blcoky pattern (high charge segregation):
    put all positive charges first, then all negative charges.
    In the SCD definition, this pattern yields more negative SCD (smaller value).
    """
    charged = extract_charged(seq)
    positives = [a for a in charged if a in POS_SET]
    negatives = [a for a in charged if a in NEG_SET]
    # positive charges first, negative charges later
    return positives + negatives

def make_alternating_pattern(seq):
    """
    alternating pattern (low charge segregation / more mixed):
    try to alternate + and - charges at charged positions.
    In the SCD definition, this pattern yields SCD closer to 0 (larger value, less negative).
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
    random pattern: randomly shuffle all charged residues
    """
    charged = extract_charged(seq)
    rng.shuffle(charged)
    return charged


# =========================
# 3. main function: generate SCD panel
# =========================

def generate_scd_panel(seq,
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
        all_df   : DataFrame with all generated mutants and SCD values
        panel_df : DataFrame (panel) uniformly sampling indices
                   from most negative SCD (most blocky) to highest SCD (most mixed)
    """
    rng = random.Random(seed)
    charged_indices = get_charged_indices(seq)
    n_charged = len(charged_indices)
    print(f"Sequence length = {len(seq)}, charged residues = {n_charged}")

    records = []

    # 1) WT
    scd_wt = compute_scd(seq)
    records.append({
        "mutant_id": "WT",
        "seq": seq,
        "scd": scd_wt,
        "type": "WT"
    })

    # 2) blocky (highly segregated -> low SCD)
    blocky_pattern = make_blocky_pattern(seq)
    seq_blocky = make_mutant_from_pattern(seq, blocky_pattern)
    scd_blocky = compute_scd(seq_blocky)
    records.append({
        "mutant_id": "blocky",
        "seq": seq_blocky,
        "scd": scd_blocky,
        "type": "designed_blocky_low_scd"
    })

    # 3) alternating (most mixed -> SCD closer to 0 / higher SCD)
    alt_pattern = make_alternating_pattern(seq)
    seq_alt = make_mutant_from_pattern(seq, alt_pattern)
    scd_alt = compute_scd(seq_alt)
    records.append({
        "mutant_id": "alternating",
        "seq": seq_alt,
        "scd": scd_alt,
        "type": "designed_mixed_high_scd"
    })

    # 4) multiple random scrambles
    # seen_seqs = {seq, seq_blocky, seq_alt}
    seen_seqs = set()
    for i in range(n_random):
        pattern = make_random_pattern(seq, rng)
        seq_rand = make_mutant_from_pattern(seq, pattern)
        if seq_rand in seen_seqs:
            continue
        seen_seqs.add(seq_rand)
        scd = compute_scd(seq_rand)
        records.append({
            "mutant_id": f"rand_{i}",
            "seq": seq_rand,
            "scd": scd,
            "type": "random"
        })

    df = pd.DataFrame(records)
    # Sort by SCD from low to high:
    #   Front: most negative SCD (most blocky)
    #   End: highest SCD (most mixed)
    df = df.sort_values("scd").reset_index(drop=True)

    # 5) Uniformly sample panel_size sequences from most negative to highest SCD (approximately uniform by index)
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
    # seq = "EDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFG"

    # all_df, panel_df = generate_scd_panel(
    #     seq,
    #     n_random=10000,   # You can increase/decrease this
    #     panel_size=1000,   # Desired panel size
    #     seed=42
    # )

    # print("\n=== All mutants summary (first 10) ===")
    # print(all_df.head(10))

    # print("\n=== SCD panel (most negative → most mixed) ===")
    # print(panel_df[["panel_rank", "mutant_id", "scd", "type"]])

    # # Save to CSV for easy use in experimental design
    # all_df.to_csv("cell_fm/tasks/cell_fm_cs/data/DDX4/all_mutants_scd.csv", index=False)
    # panel_df.to_csv("cell_fm/tasks/cell_fm_cs/data/DDX4/scd_panel.csv", index=False)

    seq = "EK" * 33

    all_df, panel_df = generate_scd_panel(
        seq,
        n_random=10000,   # You can increase/decrease this
        panel_size=1000,   # Desired panel size
        seed=42
    )

    print("\n=== All mutants summary (first 10) ===")
    print(all_df.head(10))

    print("\n=== SCD panel (most negative → most mixed) ===")
    print(panel_df[["panel_rank", "mutant_id", "scd", "type"]])

    # Save to CSV for easy use in experimental design
    all_df.to_csv("cell_fm/tasks/cell_fm_cs/data/E-K/all_mutants_scd.csv", index=False)
    panel_df.to_csv("cell_fm/tasks/cell_fm_cs/data/E-K/scd_panel.csv", index=False)