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
import pandas as pd


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

    ###################################################################################################
    # EK sequences
    protein_name = "E-K-5"

    seq_dir["EK_1"] = "EK" * 33
    seq_dir["EK_2"] = (("E" * 3 + "K" * 3) * 11)
    seq_dir["EK_3"] = (("E" * 6 + "K" * 6) * 5) + "EEEKKK"
    seq_dir["EK_4"] = (("E" * 8 + "K" * 8) * 4) + "EK"
    seq_dir["EK_5"] = (("E" * 11 + "K" * 11) * 3)
    seq_dir["EK_6"] = (("E" * 16 + "K" * 16) * 2) + "EK"
    seq_dir["EK_7"] = "E" * 15 + "KKK" + "E" * 15 + "K" * 15 + "EEE" + "K" * 15
    seq_dir["EK_8"] = "E" * 16 + "K" + "E" * 16 + "K" * 16 + "E" + "K" * 16
    seq_dir["EK_9"] = "E" * 33 + "K" * 33
    
    ###################################################################################################

    ###################################################################################################
    # # EKQ sequences
    # protein_name = "E-K-Q"

    # seq_dir["EK_1"] = "EK" * 30
    # seq_dir["EK_2"] = (("E" * 3 + "K" * 3) * 10)
    # seq_dir["EK_3"] = (("E" * 5 + "K" * 5) * 6)
    # seq_dir["EK_4"] = (("E" * 10 + "K" * 10) * 3)
    # seq_dir["EK_5"] = (("E" * 15 + "K" * 15) * 2)
    # seq_dir["EK_6"] = "E" * 14 + "KK" + "E" * 14 + "K" * 14 + "EE" + "K" * 14
    # seq_dir["EK_7"] = "E" * 30 + "K" * 30

    # # evenly distribute Q in the sequence, total length 66
    # for key in list(seq_dir.keys()):
    #     seq = seq_dir[key]
    #     if len(seq) < 66:
    #         neutral_aa = "Q"
    #         num_to_add = 66 - len(seq)
    #         interval = len(seq) // (num_to_add + 1)
    #         new_seq = ""
    #         q_added = 0
    #         for i, aa in enumerate(seq):
    #             if (i + 1) % interval == 0 and q_added < num_to_add:
    #                 new_seq += neutral_aa
    #                 q_added += 1
    #             new_seq += aa
    #         # if still have Q to add, append at the end
    #         while q_added < num_to_add:
    #             new_seq += neutral_aa
    #             q_added += 1
    #         seq_dir[key] = new_seq
    
    ###################################################################################################

    ###################################################################################################
    # # DK sequences
    # protein_name = "D-K"

    # seq_dir["DK_1"] = "DK" * 33
    # seq_dir["DK_2"] = (("D" * 3 + "K" * 3) * 11)
    # seq_dir["DK_3"] = (("D" * 6 + "K" * 6) * 5) + "DDDKKK"
    # seq_dir["DK_4"] = (("D" * 8 + "K" * 8) * 4) + "DK"
    # seq_dir["DK_5"] = (("D" * 11 + "K" * 11) * 3)
    # seq_dir["DK_6"] = (("D" * 16 + "K" * 16) * 2) + "DK"
    # seq_dir["DK_7"] = "D" * 15 + "KKK" + "D" * 15 + "K" * 15 + "DDD" + "K" * 15
    # seq_dir["DK_8"] = "D" * 16 + "K" + "D" * 16 + "K" * 16 + "D" + "K" * 16
    # seq_dir["DK_9"] = "D" * 33 + "K" * 33

    ###################################################################################################

    ###################################################################################################
    # # EK sequences
    # protein_name = "E-K"

    # seq_dir["EK_1"] = "EK" * 33
    # seq_dir["EK_2"] = (("E" * 3 + "K" * 3) * 11)
    # seq_dir["EK_3"] = (("E" * 6 + "K" * 6) * 5) + "EEEKKK"
    # seq_dir["EK_4"] = (("E" * 8 + "K" * 8) * 4) + "EK"
    # seq_dir["EK_5"] = (("E" * 11 + "K" * 11) * 3)
    # seq_dir["EK_6"] = (("E" * 16 + "K" * 16) * 2) + "EK"
    # seq_dir["EK_7"] = "E" * 15 + "KKK" + "E" * 15 + "K" * 15 + "EEE" + "K" * 15
    # seq_dir["EK_8"] = "E" * 16 + "K" + "E" * 16 + "K" * 16 + "E" + "K" * 16
    # seq_dir["EK_9"] = "E" * 33 + "K" * 33
    
    ###################################################################################################

    ###################################################################################################
    # EK3 sequences
    # protein_name = "E-K-3"

    # seq_dir["EK_1"] = "EK" * 33
    # seq_dir["EK_2"] = (("E" * 11 + "K" * 11) * 3)
    # seq_dir["EK_3"] = (("E" * 16 + "K" * 16) * 2) + "EK"
    # seq_dir["EK_4"] = "E" * 16 + "K" + "E" * 16 + "K" * 16 + "E" + "K" * 16
    # seq_dir["EK_5"] = "E" * 33 + "K" * 33
    
    ###################################################################################################

    ###################################################################################################
    # # EK3 sequences
    # protein_name = "E-K-4"

    # seq_dir["EK_1"] = "EK" * 33
    # seq_dir["EK_2"] = (("E" * 11 + "K" * 11) * 3)
    # seq_dir["EK_3"] = (("E" * 16 + "K" * 16) * 2) + "EK"
    # seq_dir["EK_4"] = "E" * 16 + "K" + "E" * 16 + "K" * 16 + "E" + "K" * 16
    # seq_dir["EK_5"] = "E" * 33 + "K" * 33
    
    ###################################################################################################

    ###################################################################################################

    # # ER sequences
    # protein_name = "E-R"
    # totle_amino_acids = 66

    # seq_dir["ER_1"] = "ER" * 15
    # seq_dir["ER_2"] = (("E" * 3 + "R" * 3) * 5)
    # seq_dir["ER_3"] = (("E" * 5 + "R" * 5) * 3) 
    # seq_dir["ER_4"] = (("E" * 7 + "R" * 7) * 2) + "ER"
    # seq_dir["ER_5"] = "E" * 15 + "R" * 15

    # # padd with neutral amino acids to make length 66
    # for key in list(seq_dir.keys()):
    #     seq = seq_dir[key]
    #     if len(seq) < totle_amino_acids:
    #         neutral_aa = "G"
    #         num_to_add = totle_amino_acids - len(seq)
    #         seq = seq + neutral_aa * num_to_add
    #         seq_dir[key] = seq

    ###################################################################################################

    ###################################################################################################

    # # EK2 sequences
    # protein_name = "E-K-2"
    # totle_amino_acids = 66

    # seq_dir["EK_1"] = "EK" * 20
    # seq_dir["EK_2"] = (("E" * 5 + "K" * 5) * 4)
    # seq_dir["EK_3"] = (("E" * 10 + "K" * 10) * 2)
    # seq_dir["EK_4"] = "E" * 9 + "KK" + "E" * 9 + "K" * 9 + "EE" + "K" * 9
    # seq_dir["EK_5"] = "E" * 20 + "K" * 20

    # # padd with neutral amino acids to make length 66
    # for key in list(seq_dir.keys()):
    #     seq = seq_dir[key]
    #     if len(seq) < totle_amino_acids:
    #         neutral_aa = "G"
    #         num_to_add = totle_amino_acids - len(seq)
    #         seq = seq + neutral_aa * num_to_add
    #         seq_dir[key] = seq

    ###################################################################################################

    ###################################################################################################
    # # DDX4 sequences
    # protein_name = "DDX4"
    # seq = "EDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFG"

    # seq_dir["DDX4_WT"] = seq

    # from generate_mutation_kappa import make_alternating_pattern, make_blocky_pattern, make_mutant_from_pattern

    # # blocky (highly segregated)
    # blocky_pattern = make_blocky_pattern(seq)
    # seq_blocky = make_mutant_from_pattern(seq, blocky_pattern)

    # seq_dir["DDX4_blocky"] = seq_blocky

    # # alternating (well-mixed)
    # alt_pattern = make_alternating_pattern(seq)
    # seq_alt = make_mutant_from_pattern(seq, alt_pattern)

    # seq_dir["DDX4_alternating"] = seq_alt

    # # positive charge to nagative charge mutation (R/K/H to E)
    # seq_dir['DDX4_pos_to_neg'] = seq.replace('R', 'E').replace('K', 'E').replace('H', 'E')
    ###################################################################################################

    ###################################################################################################
    # # LAF-1 sequences
    # protein_name = "LAF-1"
    # seq = "GRYVPPHLRGGDGGAAAAASAGGDDRRGGAGGGGYRRGGGNSGGGGGGGYDRGYNDNRDDRDNRGG"

    # seq_dir[f"{protein_name}_WT"] = seq

    # from generate_mutation_kappa import make_alternating_pattern, make_blocky_pattern, make_mutant_from_pattern

    # # blocky (highly segregated)
    # blocky_pattern = make_blocky_pattern(seq)
    # seq_blocky = make_mutant_from_pattern(seq, blocky_pattern)

    # seq_dir[f"{protein_name}_blocky"] = seq_blocky

    # # alternating (well-mixed)
    # alt_pattern = make_alternating_pattern(seq)
    # seq_alt = make_mutant_from_pattern(seq, alt_pattern)

    # seq_dir[f"{protein_name}_alternating"] = seq_alt

    # # positive charge to nagative charge mutation (R/K/H to E)
    # seq_dir[f'{protein_name}_pos_to_neg'] = seq.replace('R', 'E').replace('K', 'E').replace('H', 'E')
    ###################################################################################################

    ###################################################################################################
    # # LAF-1_2 sequences
    # protein_name = "LAF-1_2"
    # seq = "DWLEGMSGDMRSGGGYRGRGGRGNGQRFGGRDHRYQGGSGNGGGGNGGGGGFGGGGQRSGGGGGFQ"

    # seq_dir[f"{protein_name}_WT"] = seq

    # from generate_mutation_kappa import make_alternating_pattern, make_blocky_pattern, make_mutant_from_pattern

    # # blocky (highly segregated)
    # blocky_pattern = make_blocky_pattern(seq)
    # seq_blocky = make_mutant_from_pattern(seq, blocky_pattern)

    # seq_dir[f"{protein_name}_blocky"] = seq_blocky

    # # alternating (well-mixed)
    # alt_pattern = make_alternating_pattern(seq)
    # seq_alt = make_mutant_from_pattern(seq, alt_pattern)

    # seq_dir[f"{protein_name}_alternating"] = seq_alt

    # # positive charge to nagative charge mutation (R/K/H to E)
    # seq_dir[f'{protein_name}_pos_to_neg'] = seq.replace('R', 'E').replace('K', 'E').replace('H', 'E')
    ###################################################################################################

    ###################################################################################################
    # protein_name = "hnRNPA1-2"
    # seq_dir[protein_name] = "NGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGG"
    ###################################################################################################

    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    from generate_mutation_kappa import compute_kappa
    from generate_mutation_scd import compute_scd

    kappas = []
    scds = []

    for key in seq_dir.keys():
        seq = seq_dir[key]
        kappa = compute_kappa(seq)
        scd = compute_scd(seq)

        kappas.append(kappa)
        scds.append(scd)

    # save seq_dir, kappa, scd to csv file
    df = pd.DataFrame({
        "mutant": list(seq_dir.keys()),
        "sequence": list(seq_dir.values()),
        "kappa": kappas,
        "scd": scds,
    })

    save_path = f'cell_fm/tasks/cell_fm_cs/data/{protein_name}/'
    os.makedirs(save_path, exist_ok=True)

    df.to_csv(os.path.join(save_path, 'panel.csv'), index=False)
    # assert False 

    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    print("All sequences to check:")
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