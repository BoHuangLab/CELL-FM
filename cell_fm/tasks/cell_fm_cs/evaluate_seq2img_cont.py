# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.condenseq_data.dataset import CondenSeqAllImageDataset
from cell_fm.models.cell_fm.cell_fm_cs_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_cs_model import CELLFMCSModel
from cell_fm.utils.cli_utils import cli
from cell_fm.logging import logger

import tifffile as tiff
import numpy as np

from tqdm import tqdm
from esm.utils import encoding
import pandas as pd

from cell_fm.pipeline.accelerator.trainer import seed_everything


def save_tif(image, output_path):
    # image = np.transpose(image, (1, 0, 2, 3))
    image = image * 65535

    image = image.astype(np.uint16)
    tiff.imwrite(output_path, image, imagej=True)


@cli(CELLFMConfig)
def main(args) -> None:    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))

    model = CELLFMCSModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    seq_dir = {}

    # hnRNPA1 sequences
    # protein_name = "hnRNPA1"
    # seq_dir[protein_name] = "NGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGG"

    # for key in [protein_name]:
    #     orig_seq = seq_dir[key]

    #     # find all Y positions in the original sequence (0-based index)
    #     y_positions = [i for i, aa in enumerate(orig_seq) if aa == "Y"]  # e.g. [8, 15, 20, ...]
    #     # construct mutants by changing first k Y to F (1 <= k <= len(y_positions))
    #     for k in range(1, len(y_positions) + 1):
    #         # based on the original sequence, make a copy as a list that can be modified
    #         seq_list = list(orig_seq)
    #         # change first k Y to F
    #         for pos in y_positions[:k]:
    #             seq_list[pos] = "F"
    #         mutant_seq = "".join(seq_list)

    #         # generate labels like Y9F, Y9F_Y16F, Y9F_Y16F_Y23F
    #         labels = [f"Y{pos+1}F" for pos in y_positions[:k]]  # +1 to convert to 1-based indexing
    #         mut_key = key + "_" + "_".join(labels)

    #         seq_dir[mut_key] = mutant_seq

    # # NUP98 sequences
    # protein_name = "NUP98"
    # seq_dir[protein_name] = "KSFGTPFGGGTGGFGTTSTFGQNTGFGTTSGGAFGTSAFGSSNNTGGLFGNSQTKPGGLFGTSSFS"

    # for key in [protein_name]:
    #     orig_seq = seq_dir[key]

    #     # find all F positions in the original sequence (0-based index)
    #     f_positions = [i for i, aa in enumerate(orig_seq) if aa == "F"]  # e.g. [8, 15, 20, ...]
    #     # construct mutants by changing first k F to Y (1 <= k <= len(f_positions))
    #     for k in range(1, len(f_positions) + 1):
    #         # based on the original sequence, make a copy as a list that can be modified
    #         seq_list = list(orig_seq)
    #         # change first k F to S
    #         for pos in f_positions[:k]:
    #             seq_list[pos] = "S"
    #         mutant_seq = "".join(seq_list)

    #         # generate labels。
    #         labels = [f"F{pos+1}S" for pos in f_positions[:k]]  # +1 to convert to 1-based indexing
    #         mut_key = key + "_" + "_".join(labels)

    #         seq_dir[mut_key] = mutant_seq

    # for key in ["6179"]:
    #     orig_seq = seq_dir[key]

    #     # find all F positions in the original sequence (0-based index)
    #     f_positions = [i for i, aa in enumerate(orig_seq) if aa == "F"]  # e.g. [8, 15, 20, ...]
    #     # construct mutants by changing first k F to S (1 <= k <= len(f_positions))
    #     for k in range(1, len(f_positions) + 1):
    #         # based on the original sequence, make a copy as a list that can be modified
    #         seq_list = list(orig_seq)
    #         # change k-th F to S
    #         pos = f_positions[k-1]
    #         seq_list[pos] = "S"
    #         mutant_seq = "".join(seq_list)

    #         # generate labels.
    #         labels = f"F{pos+1}S"  # +1 to convert to 1-based indexing
    #         mut_key = key + "_" + labels

    #         seq_dir[mut_key] = mutant_seq

    # for key in ["6179"]:
    #     orig_seq = seq_dir[key]

    #     # find all F positions in the original sequence (0-based index)
    #     f_positions = [i for i, aa in enumerate(orig_seq) if aa == "F"]  # e.g. [8, 15, 20, ...]
    #     # construct mutants by changing first k F to A (1 <= k <= len(f_positions))
    #     for k in range(1, len(f_positions) + 1):
    #         # based on the original sequence, make a copy as a list that can be modified
    #         seq_list = list(orig_seq)
    #         # change first k F to A
    #         for pos in f_positions[:k]:
    #             seq_list[pos] = "A"
    #         mutant_seq = "".join(seq_list)

    #         # generate labels.
    #         labels = [f"F{pos+1}A" for pos in f_positions[:k]]  # +1 to convert to 1-based indexing
    #         mut_key = key + "_" + "_".join(labels)

    #         seq_dir[mut_key] = mutant_seq

    # for key in ["6179"]:
    #     orig_seq = seq_dir[key]

    #     # 1. Find all FG motif positions (record the index of F, 0-based)
    #     fg_positions = [
    #         i for i in range(len(orig_seq) - 1)
    #         if orig_seq[i] == "F" and orig_seq[i + 1] == "G"
    #     ]  # e.g. [8, 15, 20, ...]

    #     # 2. Construct mutants by changing first k FG to FE (only change G to E) (1 <= k <= len(fg_positions))
    #     for k in range(1, len(fg_positions) + 1):
    #         # Based on the original sequence, make a copy as a list that can be modified
    #         seq_list = list(orig_seq)

    #         # Change the G in the first k FG motifs to E
    #         for pos in fg_positions[:k]:
    #             seq_list[pos + 1] = "E"  # pos is the index of F, pos+1 is the index of G

    #         mutant_seq = "".join(seq_list)

    #         # Generate labels, e.g., G10E (G at 1-based position 10 changed to E)
    #         labels = [f"G{pos + 2}E" for pos in fg_positions[:k]]  # pos+1 is G's 0-based index, +1 to convert to 1-based
    #         mut_key = key + "_" + "_".join(labels)

    #         # Save mutant sequences
    #         seq_dir[mut_key] = mutant_seq

    # FUS sequences
    # seq_dir["13629"] = "TSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGG"
    # seq_dir["13629_G48E"] = "TSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYEQQNQYNSSSGGGGGGGGG"

    # DDX4 sequences
    # protein_name = "DDX4"
    # df = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/DDX4/scd_panel.csv")

    # for _, row in df.iterrows():
    #     seq_dir['DDX4' + "_" + str(row['mutant_id'])] = row['seq']

    # ELP sequences
    protein_name = "ELP"
    AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for aa in AAs:
        seq_dir[f"ELP_X={aa}"] = (f"VPG{aa}G" * 14)[:66]

    # print all sequences to check
    for k, v in seq_dir.items():
        print(k, ":", v)

    # chosen images
    protein_index = 12626
    index = 0

    valset = CondenSeqAllImageDataset(args, split_key='all')
    valset.meta_data = valset.meta_data[valset.meta_data['index'] == protein_index]
    chosen_data = valset.__getitem__(0)

    vocab = valset.vocab

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)

    chosen_cell_img = chosen_nucleus_img

    # max_protein_intensity_level = 4096
    # num_protein_intensity_levels = 4096
    # batch_size = 256

    max_protein_intensity_level = 800
    num_protein_intensity_levels = 4096
    batch_size = 256

    protein_intensity_levels_linspace = torch.linspace(0, max_protein_intensity_level, steps=num_protein_intensity_levels).to(device)

    for mut_key, protein_seq in tqdm(seq_dir.items()):
        # seed_everything(config.seed)

        output_folder_name = f'protein_{mut_key}'

        output_dir = Path(config.output_dir)
        output_dir = output_dir / protein_name / output_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(protein_seq)

        protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)

        generated_imgs = []

        for batch_start in range(0, num_protein_intensity_levels, batch_size):
            batch_end = min(batch_start + batch_size, num_protein_intensity_levels)
            current_batch_size = batch_end - batch_start

            protein_intensity_level_batch = protein_intensity_levels_linspace[batch_start:batch_end]
            protein_intensity_level_batch = protein_intensity_level_batch.unsqueeze(1).to(device)
            protein_seq_batch = protein_seq.repeat(current_batch_size, 1).to(device)
            chosen_cell_img_batch = chosen_cell_img.repeat(current_batch_size, 1, 1, 1).to(device)

            # Generate images
            sample = model.sequence_to_image(
                protein_seq_batch, 
                chosen_cell_img_batch, 
                protein_intensity_level_batch, 
                num_steps=args.num_steps, 
            )
            generated_imgs.append(sample)

        generated_imgs = torch.cat(generated_imgs, dim=0)

        # rescale images to [0, 1]
        combined_generated_imgs = torch.cat([chosen_nucleus_img.repeat(generated_imgs.size(0), 1, 1, 1), generated_imgs], dim=1)  # [N, 2, H, W]
        combined_generated_imgs = (combined_generated_imgs + 1) / 2.0  # shift to [0,1]
        combined_generated_imgs = combined_generated_imgs.clamp(0.0, 1.0)

        save_tif(
            combined_generated_imgs.cpu().numpy(),
            output_dir / "combined_images.tif", 
        )

        # save protein_intensity_levels_linspace as npy
        np.save(
            output_dir / "protein_intensity_levels.npy",
            protein_intensity_levels_linspace.cpu().numpy(),
        )

    logger.info("Done!")

if __name__ == "__main__":
    main()