import pandas as pd

result1_df = pd.read_csv('/home/dihan.zheng/workspace/dihan.zheng/dataset/human_protein/CALVADOS2_hp_idr_PSprediction_with_names.csv')

# convert 'Delta G [kT]' column to numeric, coerce errors to NaN
result1_df['Delta G [kT]'] = pd.to_numeric(result1_df['Delta G [kT]'], errors='coerce')

# sort by 'Delta G [kT]' ascending
result1_df = result1_df.sort_values(by='Delta G [kT]', ascending=True).reset_index(drop=True)

result2_df = pd.read_csv('output/condenseq/blind_screening_hp_idr/human_idr_regions_plddt_only_chopped_unique_seq_with_predicted_fraction.csv')

# sort by 'predicted_fraction' descending
result2_df = result2_df.sort_values(by='predicted_fraction', ascending=False).reset_index(drop=True)

# ===========================
# 1. Establish "rankings" for each protein_names
# ===========================

# for result1：same protein_names may appear multiple times, here only keep the first occurrence (highest rank)
r1_rank = (
    result1_df[['Sequence']]
    .drop_duplicates()
    .reset_index(drop=True)
)
r1_rank['rank_deltaG'] = r1_rank.index + 1   # ranking starts from 1

# for result2, similarly
r2_rank = (
    result2_df[['sequence']]
    .drop_duplicates()
    .reset_index(drop=True)
)
r2_rank['rank_pred_frac'] = r2_rank.index + 1   # ranking starts from 1

# only consider sequences that appear in both results
common = pd.merge(r1_rank, r2_rank, left_on='Sequence', right_on='sequence', how='inner')

print(f"Number of common sequences in both results: {len(common)}")

# ===========================
# 2. Spearman rank correlation
# ===========================

spearman_corr = common[['rank_deltaG', 'rank_pred_frac']].corr(method='spearman').loc['rank_deltaG', 'rank_pred_frac']
print(f"Spearman rank correlation between rankings = {spearman_corr:.3f}")

# ===========================
# 3. top-K overlap analysis
# ===========================

def topk_overlap(k):
    top1 = set(r1_rank.head(k)['Sequence'])
    top2 = set(r2_rank.head(k)['sequence'])
    overlap = top1 & top2
    return len(overlap), overlap

for k in [20, 50, 100, 200, 500, 2000]:
    n_overlap, _ = topk_overlap(k)
    print(f"Top-{k} overlap: {n_overlap} sequence (fraction = {n_overlap / k:.3f})")