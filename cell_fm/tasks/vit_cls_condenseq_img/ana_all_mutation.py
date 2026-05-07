from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import matplotlib.colors as colors


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

        mt_count = pd.read_csv(os.path.join(mt_path, 'pred_vs_intensity.csv'))
        wt_count = pd.read_csv(os.path.join(wt_path, 'pred_vs_intensity.csv'))

        mt_pred = (mt_count['predicted_class'] == 1).sum() / len(mt_count)
        wt_pred = (wt_count['predicted_class'] == 1).sum() / len(wt_count)

        diff = (mt_pred - wt_pred) / wt_pred if wt_pred != 0 else 0.0
        # diff = mt_pred - wt_pred
        all_diffs.append(diff)

    if len(all_diffs) > 0:
        mean_val = np.mean(all_diffs)
        median_val = np.median(all_diffs)
        var_val = np.var(all_diffs, ddof=1) if len(all_diffs) > 1 else 0.0
    else:
        mean_val = np.nan
        median_val = np.nan
        var_val = np.nan

    return mutation_type, median_val, var_val, mean_val


def main() -> None:
    # save_path = Path('output/condenseq/analysis/mutation_impact')
    save_path = Path('output/condenseq/analysis/mutation_impact_4096_512')
    # save_path = Path('output/condenseq/analysis/mutation_impact_4096_512_for_reentrant')
    save_path.mkdir(parents=True, exist_ok=True)

    # data_path = '/hpc/reference/opencell/condenseq/seq2img_all_mutation/PT_CondenSeq_CELL-Diff2_Rep_Dev_GFP_e_4_ignl_4_ignah_8_S1_R1_50k/cellfm_test_selected_balanced'
    data_path = '/hpc/reference/opencell/condenseq/seq2img_all_mutation_reentrant/cellfm_test_selected_balanced'
    # data_path = '/hpc/reference/opencell/condenseq/seq2img_all_mutation_reentrant/cellfm_train_selected_balanced_for_reentrant'
    
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
    plt.figure(figsize=(8, 7), facecolor='none')  # figure 背景透明

    norm = colors.TwoSlopeNorm(vcenter=0, vmin=np.min(median_mat), vmax=np.max(median_mat))
    im = plt.imshow(median_mat, origin='upper', cmap='bwr', norm=norm)

    ax = plt.gca()
    ax.set_facecolor('none')  # axes 背景透明

    cbar = plt.colorbar(im)
    cbar.ax.set_facecolor('none')  # colorbar 背景透明

    plt.xticks(np.arange(n_aa), AA)
    plt.yticks(np.arange(n_aa), AA)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    plt.xlabel('Mutate TO', fontsize=18)
    plt.ylabel('Mutate FROM', fontsize=18)
    plt.title('Median Δ predicted condensate fraction', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        save_path / 'mutation_median_matrix.svg',
        dpi=300, bbox_inches='tight', pad_inches=0.0,
        transparent=True  # 保存为透明背景
    )
    plt.close()

    # =======================
    #  plot variance matrix (no need for 0=white)
    # =======================
    plt.figure(figsize=(8, 7), facecolor='none')

    im = plt.imshow(var_mat, origin='upper', cmap='viridis')

    ax = plt.gca()
    ax.set_facecolor('none')

    cbar = plt.colorbar(im)
    cbar.ax.set_facecolor('none')

    plt.xticks(np.arange(n_aa), AA)
    plt.yticks(np.arange(n_aa), AA)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    plt.xlabel('Mutate TO', fontsize=18)
    plt.ylabel('Mutate FROM', fontsize=18)
    plt.title('Variance of Δ predicted condensate fraction', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        save_path / 'mutation_variance_matrix.svg',
        dpi=300, bbox_inches='tight', pad_inches=0.0,
        transparent=True
    )
    plt.close()

    # =======================
    #  plot mean matrix
    # =======================
    plt.figure(figsize=(8, 7), facecolor='none')

    norm = colors.TwoSlopeNorm(vcenter=0, vmin=np.min(mean_mat), vmax=np.max(mean_mat))
    im = plt.imshow(mean_mat, origin='upper', cmap='bwr', norm=norm)

    ax = plt.gca()
    ax.set_facecolor('none')

    cbar = plt.colorbar(im)
    cbar.ax.set_facecolor('none')

    plt.xticks(np.arange(n_aa), AA)
    plt.yticks(np.arange(n_aa), AA)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    plt.xlabel('Mutate TO', fontsize=18)
    plt.ylabel('Mutate FROM', fontsize=18)
    plt.title('Mean Δ predicted condensate fraction', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        save_path / 'mutation_mean_matrix.svg',
        dpi=300, bbox_inches='tight', pad_inches=0.0,
        transparent=True
    )
    plt.close()

    # save mutation_median_results and mutation_var_results as CSV
    # each row: amino acid in AA
    # each column: amino acid in AA
    # from row to column: mutation_type
    median_df = pd.DataFrame(median_mat, index=AA, columns=AA)
    var_df = pd.DataFrame(var_mat, index=AA, columns=AA)
    mean_df = pd.DataFrame(mean_mat, index=AA, columns=AA)

    median_df.to_csv(save_path / 'mutation_median_matrix.csv')
    var_df.to_csv(save_path / 'mutation_variance_matrix.csv')
    mean_df.to_csv(save_path / 'mutation_mean_matrix.csv')

    print(f'Saved results to {save_path}')

if __name__ == "__main__":
    main()
