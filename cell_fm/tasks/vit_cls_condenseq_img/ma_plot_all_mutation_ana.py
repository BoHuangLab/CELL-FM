from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import matplotlib.colors as colors
import json


AA = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C',
      'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']


def compute_mutation_stats(args):
    """
    for single mutation_type:
    - compute all_diffs median
    - then compute variance
    """
    mutation_type, all_wt_data_paths, data_path = args

    all_data_paths = sorted(glob(os.path.join(data_path, mutation_type, '*')))
    all_diffs = []

    for mt_path, wt_path in zip(all_data_paths, all_wt_data_paths):
        mt_protein_seq = np.loadtxt(os.path.join(mt_path, 'protein_seq.txt'), dtype=str)
        wt_protein_seq = np.loadtxt(os.path.join(wt_path, 'protein_seq.txt'), dtype=str)

        if mt_protein_seq == wt_protein_seq:
            continue

        mt_result = json.load(open(os.path.join(mt_path, 'predicted_condensate_probability_ma_mgap.json'), 'r'))
        wt_result = json.load(open(os.path.join(wt_path, 'predicted_condensate_probability_ma_mgap.json'), 'r'))

        mt_max_gap = mt_result.get('max_gap', None)
        wt_max_gap = wt_result.get('max_gap', None)

        # diff = (mt_max_gap - wt_max_gap) / wt_max_gap if wt_max_gap != 0 else 0.0
        diff = mt_max_gap - wt_max_gap
        all_diffs.append(diff)

    if len(all_diffs) > 0:
        mean_val = np.mean(all_diffs)
        median_val = np.median(all_diffs)
        var_val = np.var(all_diffs, ddof=1) if len(all_diffs) > 1 else 0.0
    else:
        mean_val = np.nan
        median_val = np.nan
        var_val = np.nan
        print(f'No valid data for mutation type {mutation_type}')

    return mutation_type, median_val, var_val, mean_val


def main() -> None:
    save_path = Path('output/condenseq/analysis/max_gap_all_mutation/cellfm_test_selected_balanced')
    save_path.mkdir(parents=True, exist_ok=True)

    # data_path = '/hpc/reference/opencell/condenseq/seq2img_all_mutation_reentrant/cellfm_train_selected_balanced_for_reentrant'
    data_path = '/hpc/reference/opencell/condenseq/seq2img_all_mutation_reentrant/cellfm_test_selected_balanced'
    
    # WT
    all_wt_data_paths = sorted(glob(os.path.join(data_path, 'WT', '*')))

    # all mutation types: e.g., 'R2H', 'R2K', ..., 'W2Y', 'W2F'
    all_mutation_types = [f'{aa1}2{aa2}' for aa1 in AA for aa2 in AA if aa1 != aa2]

    tasks = [(mutation_type, all_wt_data_paths, data_path) for mutation_type in all_mutation_types]

    mutation_median_results = {}
    mutation_var_results = {}
    mutation_mean_results = {}

    n_workers = cpu_count()
    print(f'Using {n_workers} workers')

    with Pool(processes=n_workers) as pool:
        for mut_type, median_val, var_val, mean_val in tqdm(
            pool.imap_unordered(compute_mutation_stats, tasks),
            total=len(tasks),
            desc='Processing all mutations'
        ):
            mutation_median_results[mut_type] = median_val
            mutation_var_results[mut_type] = var_val
            mutation_mean_results[mut_type] = mean_val

    # =======================
    # summarize to matrices
    # =======================
    n_aa = len(AA)
    median_mat = np.full((n_aa, n_aa), np.nan, dtype=float)
    var_mat = np.full((n_aa, n_aa), np.nan, dtype=float)
    mean_mat = np.full((n_aa, n_aa), np.nan, dtype=float)

    for i, aa1 in enumerate(AA):          # row: original amino acid (from)
        for j, aa2 in enumerate(AA):      # column: mutated amino acid (to)
            if aa1 == aa2:
                # set to zero
                median_mat[i, j] = 0.0
                var_mat[i, j] = 0.0
                mean_mat[i, j] = 0.0
                # continue  # diagonal has no corresponding mutation_type (or remains nan)
            mut_type = f'{aa1}2{aa2}'
            if mut_type in mutation_median_results:
                median_mat[i, j] = mutation_median_results[mut_type]
            if mut_type in mutation_var_results:
                var_mat[i, j] = mutation_var_results[mut_type]
            if mut_type in mutation_mean_results:
                mean_mat[i, j] = mutation_mean_results[mut_type]

    # =======================
    #  plot median matrix
    # =======================
    plt.figure(figsize=(8, 7))

    # Use TwoSlopeNorm to ensure 0 is white
    norm = colors.TwoSlopeNorm(vcenter=0, vmin=np.min(median_mat), vmax=np.max(median_mat))

    print(np.min(median_mat), np.max(median_mat))

    im = plt.imshow(median_mat, origin='upper', cmap='bwr', norm=norm)
    plt.colorbar(im, label='Median (mt_max_gap - wt_max_gap)')
    plt.xticks(np.arange(n_aa), AA)
    plt.yticks(np.arange(n_aa), AA)
    plt.xlabel('Mutate TO')
    plt.ylabel('Mutate FROM')
    plt.title('Median Δ Max Gap')
    plt.tight_layout()
    plt.savefig(save_path / 'mutation_max_gap_median_matrix.png', dpi=300)
    plt.close()

    # =======================
    #  plot variance matrix (no need for 0=white)
    # =======================
    plt.figure(figsize=(8, 7))
    im = plt.imshow(var_mat, origin='upper', cmap='viridis')
    plt.colorbar(im, label='Variance of (mt_max_gap - wt_max_gap)')
    plt.xticks(np.arange(n_aa), AA)
    plt.yticks(np.arange(n_aa), AA)
    plt.xlabel('Mutate TO')
    plt.ylabel('Mutate FROM')
    plt.title('Variance of Δ Max Gap')
    plt.tight_layout()
    plt.savefig(save_path / 'mutation_max_gap_variance_matrix.png', dpi=300)
    plt.close()

    # =======================
    #  plot mean matrix
    # =======================
    plt.figure(figsize=(8, 7))

    norm = colors.TwoSlopeNorm(vcenter=0, vmin=np.min(mean_mat), vmax=np.max(mean_mat))

    print(np.min(mean_mat), np.max(mean_mat))

    im = plt.imshow(mean_mat, origin='upper', cmap='bwr', norm=norm)
    plt.colorbar(im, label='Mean (mt_max_gap - wt_max_gap)')
    plt.xticks(np.arange(n_aa), AA)
    plt.yticks(np.arange(n_aa), AA)
    plt.xlabel('Mutate TO')
    plt.ylabel('Mutate FROM')
    plt.title('Mean Δ Max Gap')
    plt.tight_layout()
    plt.savefig(save_path / 'mutation_max_gap_mean_matrix.png', dpi=300)
    plt.close()

    # save mutation_median_results and mutation_var_results as CSV
    # each row: amino acid in AA
    # each column: amino acid in AA
    # from row to column: mutation_type
    median_df = pd.DataFrame(median_mat, index=AA, columns=AA)
    var_df = pd.DataFrame(var_mat, index=AA, columns=AA)
    mean_df = pd.DataFrame(mean_mat, index=AA, columns=AA)

    median_df.to_csv(save_path / 'mutation_max_gap_median_matrix.csv')
    var_df.to_csv(save_path / 'mutation_max_gap_variance_matrix.csv')
    mean_df.to_csv(save_path / 'mutation_max_gap_mean_matrix.csv')

if __name__ == "__main__":
    main()
