import random
from itertools import combinations

protein_name = "NUP98"

sequence = 'KSFGTPFGGGTGGFGTTSTFGQNTGFGTTSGGAFGTSAFGSSNNTGGLFGNSQTKPGGLFGTSSFS'

# create mutations
seq_dir = {}
seq_dir['WT'] = sequence

# Find positions of all F residues (0-based)
f_positions = [i for i, aa in enumerate(sequence) if aa == 'F']
N = len(f_positions)

# 1 F to S: N single mutations, named 1FtoS_{1-based sequence position}
for idx, pos in enumerate(f_positions, 1):
    mutated_seq = list(sequence)
    mutated_seq[pos] = 'S'
    seq_dir[f'1FtoS_{idx}'] = ''.join(mutated_seq)

# k F to S (2 <= k <= N-1): randomly choose N combinations, named {k}FtoS_{1..N}
random.seed(42)
for k in range(2, N):
    all_combos = list(combinations(f_positions, k))
    chosen = random.sample(all_combos, min(N, len(all_combos)))
    for idx, combo in enumerate(chosen, 1):
        mutated_seq = list(sequence)
        for pos in combo:
            mutated_seq[pos] = 'S'
        seq_dir[f'{k}FtoS_{idx}'] = ''.join(mutated_seq)

# N F to S: only one mutation (all F -> S), named NFtoS_1
mutated_seq = list(sequence)
for pos in f_positions:
    mutated_seq[pos] = 'S'
seq_dir[f'{N}FtoS_1'] = ''.join(mutated_seq)

# save to csv
import pandas as pd
df = pd.DataFrame.from_dict(seq_dir, orient='index', columns=['sequence'])
df.index.name = 'mutation'
df.to_csv(f"cell_fm/tasks/cell_fm_cs/data/{protein_name}/mutation_seqs.csv")