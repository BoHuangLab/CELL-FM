import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

data_path = Path('/hpc/reference/opencell/condenseq/seq2img_blind_screening_hp_idr')

def compute_fraction(args):
    """
    input: (protein_index, data_path_str)
    output: predicted_fraction corresponding to the protein_index in pred_vs_intensity.csv
    """
    protein_index, data_path_str = args
    data_path = Path(data_path_str)
    csv_path = data_path / f'{protein_index:06d}' / 'pred_vs_intensity.csv'

    try:
        pred_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # If file not found, return NaN (you can also change to return 0 or raise)
        return float('nan')

    if len(pred_df) == 0:
        return float('nan')

    fraction = (pred_df['predicted_class'] == 1).sum() / len(pred_df)
    return fraction


def main():
    meta_data = pd.read_csv(
        "/home/dihan.zheng/workspace/dihan.zheng/dataset/human_protein/human_idr_regions_plddt_only_chopped_unique_seq_with_names.csv"
    )

    # Extract all indices
    protein_indices = meta_data['index'].tolist()

    # Prepare parameters for each task (protein_index, data_path_str)
    tasks = [(int(idx), str(data_path)) for idx in protein_indices]

    # Parallel computation
    num_workers = mp.cpu_count()  # Or set manually, e.g., 16
    fractions = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for frac in tqdm(executor.map(compute_fraction, tasks),
                         total=len(tasks),
                         desc="Computing predicted_fraction in parallel"):
            fractions.append(frac)

    # append to meta_data
    meta_data['predicted_fraction'] = fractions

    # put predicted_fraction column after index column
    cols = meta_data.columns.tolist()
    cols.insert(1, cols.pop(cols.index('predicted_fraction')))
    meta_data = meta_data[cols]

    # save to csv
    meta_data.to_csv(
        "output/condenseq/blind_screening_hp_idr/human_idr_regions_plddt_only_chopped_unique_seq_with_predicted_fraction.csv",
        index=False
    )
    print("Saved to output/condenseq/blind_screening_hp_idr/human_idr_regions_plddt_only_chopped_unique_seq_with_predicted_fraction.csv")


if __name__ == "__main__":
    main()
