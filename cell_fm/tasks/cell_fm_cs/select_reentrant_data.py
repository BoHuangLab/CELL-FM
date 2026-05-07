import os
import pandas as pd
import numpy as np
from collections import Counter
import random

df = pd.read_csv('output/condenseq/cellfm_train/cellfm_train_max_gaps.csv')
source_df = pd.read_csv('/hpc/reference/opencell/condenseq/cellfm_train_meta_data_large_pool_updated.csv')

max_gap_range = (0.2, 0.6)
y_max_minimal = 0.7

# ----------------------------
# 1) 用 df 的筛选条件找到需要的 index，然后从 source_df 取对应行
# ----------------------------

filtered_df = df[
    (df['max_gap'] >= max_gap_range[0]) &
    (df['max_gap'] <= max_gap_range[1]) &
    (df['y_max'] >= y_max_minimal)
].copy()

# 用于校验：df 里 index->protein_seq
idx_to_seq = dict(zip(filtered_df['index'], filtered_df['protein_seq']))

# 从 source_df 取对应 index 的行
filtered_source_df = source_df[source_df['index'].isin(filtered_df['index'])].copy()

# 校验：每个 index 在 source_df 里必须唯一
dup_idx = filtered_source_df['index'][filtered_source_df['index'].duplicated()].unique()
assert len(dup_idx) == 0, f"Duplicated index in source_df: {dup_idx[:10]}"

# 校验：是否有 index 在 source_df 缺失
missing = set(filtered_df['index']) - set(filtered_source_df['index'])
assert len(missing) == 0, f"Some indices not found in source_df (show up to 10): {list(missing)[:10]}"

# 校验：protein_seq 是否一致（逐行比对）
for idx, seq in idx_to_seq.items():
    src_seq = filtered_source_df.loc[filtered_source_df['index'] == idx, 'protein_seq'].values[0]
    assert src_seq == seq, f"Protein sequence mismatch for index {idx}"

# 重置索引，后面用 iloc 选子集才不会错位
filtered_source_df = filtered_source_df.reset_index(drop=True)

print('Number of selected data points:', len(filtered_source_df))

# ----------------------------
# 2) 序列列表 & AA 目标分布
# ----------------------------
sequences = filtered_source_df['protein_seq'].dropna().astype(str).tolist()
n = len(sequences)
if n == 0:
    raise ValueError("No sequences left after filtering / dropping NaNs.")

global_counter = Counter()
for seq in sequences:
    global_counter.update(seq.strip())

aas = sorted(global_counter.keys())
num_aas = len(aas)
if num_aas == 0:
    raise ValueError("No amino acids found (empty sequences?).")

# 你原来是“均匀分布目标”
global_freq = np.ones(num_aas, dtype=float) / num_aas

print("AAs:", aas)
print("Global freq (first few):", global_freq[:10])

# ----------------------------
# 3) 预计算每条序列的AA计数向量
# ----------------------------
seq_counts = np.zeros((n, num_aas), dtype=float)
for i, seq in enumerate(sequences):
    c = Counter(seq.strip())
    seq_counts[i, :] = np.array([c[a] for a in aas], dtype=float)

def compute_score(count_vec: np.ndarray) -> float:
    total = float(count_vec.sum())
    if total <= 0:
        return float("inf")
    freq = count_vec / total
    diff = freq - global_freq
    return float(np.dot(diff, diff))

# ----------------------------
# 4) hill-climbing 选 K 条
# ----------------------------
K = 100
if K > n:
    raise ValueError(f"K={K} is larger than available sequences n={n} after filtering.")
if K == n:
    print("K == n; selecting all sequences.")
    selected_indices = list(range(n))
else:
    rng = random.Random(42)
    all_indices = list(range(n))
    current_subset = set(rng.sample(all_indices, K))

    current_counts = seq_counts[list(current_subset)].sum(axis=0)
    current_score = compute_score(current_counts)
    print("Initial score:", current_score)

    num_iterations = 2000000
    subset_cache = None

    for it in range(num_iterations):
        if subset_cache is None or it % 1000 == 0:
            subset_cache = tuple(current_subset)

        i = rng.choice(subset_cache)

        # pick j outside subset
        while True:
            j = rng.randrange(n)
            if j not in current_subset:
                break

        new_counts = current_counts - seq_counts[i] + seq_counts[j]
        new_score = compute_score(new_counts)

        if new_score < current_score:
            current_subset.remove(i)
            current_subset.add(j)
            current_counts = new_counts
            current_score = new_score
            subset_cache = None

    print("Final score:", current_score)
    selected_indices = sorted(current_subset)

# ----------------------------
# 5) 导出：从 filtered_source_df 取子集（关键修改点）
# ----------------------------
selected_df = filtered_source_df.iloc[selected_indices].copy()
# selected_df.to_csv("selected_100_sequences.csv", index=False)
# print("Saved to selected_100_sequences.csv")

# ----------------------------
# 6) 验证子集AA频率
# ----------------------------
subset_counter = Counter()
for seq in selected_df['protein_seq'].dropna().astype(str).tolist():
    subset_counter.update(seq.strip())

subset_total = sum(subset_counter.values())
subset_freq = np.array([subset_counter[a] / subset_total for a in aas], dtype=float)

check_df = pd.DataFrame({
    "AminoAcid": aas,
    "GlobalFreq": global_freq,
    "SubsetFreq": subset_freq,
    "Diff": subset_freq - global_freq
}).sort_values("GlobalFreq", ascending=False)

print(check_df)

# ----------------------------
# 7) 保存到指定路径
# ----------------------------
output_path = "/hpc/reference/opencell/condenseq/cellfm_train_selected_balanced_for_reentrant_meta_data_large_pool_updated.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
selected_df.to_csv(output_path, index=False)
print(f"Saved balanced subset of 100 sequences to: {output_path}")