import pandas as pd
from cell_fm.tasks.cell_fm_cs.generate_mutation_kappa import compute_kappa
from cell_fm.tasks.cell_fm_cs.generate_mutation_scd import compute_scd

def compute_ncpr(seq):
    """
    Compute Net Charge Per Residue (NCPR).
    NCPR = (N_pos - N_neg) / L
    where:
        positive residues: R, K, H
        negative residues: D, E
    """
    pos = set(['R', 'K', 'H'])
    neg = set(['D', 'E'])

    n_pos = sum(1 for aa in seq if aa in pos)
    n_neg = sum(1 for aa in seq if aa in neg)
    L = len(seq)

    if L == 0:
        return 0.0

    return (n_pos - n_neg) / L

df = pd.read_csv('/hpc/reference/opencell/condenseq/cellfm_test_meta_data_large_pool_updated.csv')

scd_list = []
kappa_list = []
ncpr_list = []
index_list = []

for idx, row in df.iterrows():
    index = row['index']
    seq = row['protein_seq']
    kappa = compute_kappa(seq)
    scd = compute_scd(seq)

    # compute NCPR
    ncpr = compute_ncpr(seq)

    index_list.append(index)
    scd_list.append(scd)
    kappa_list.append(kappa)
    ncpr_list.append(ncpr)

# save to index, scd, kappa, ncpr to new file csv
output_df = pd.DataFrame({
    'index': index_list,
    'scd': scd_list,
    'kappa': kappa_list,
    'ncpr': ncpr_list
})

output_df.to_csv('/hpc/reference/opencell/condenseq/cellfm_test_meta_data_large_pool_updated_scd_kappa_ncpr.csv', index=False)
