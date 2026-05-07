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


def save_tif(image, output_path):
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

    # ###################################################################################################
    # seq_dir = {}
    # protein_name = "hnRNPA1"
    # seq_dir[protein_name] = "NGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGG"
    # ###################################################################################################

    ###################################################################################################
    # protein_name = "NUP98"
    # seq_dir = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/NUP98/mutation_seqs.csv", index_col=0, header=0, dtype=str).to_dict(orient='index')
    # for key in seq_dir:
    #     seq_dir[key] = seq_dir[key]['sequence']
    ###################################################################################################

    ###################################################################################################
    # protein_name = "E-K"
    # seq_dir = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/E-K/panel.csv", index_col=0, header=0, dtype=str).to_dict(orient='index')
    # # get the "mutant" column for key and "sequence" column for value
    # for key in seq_dir:
    #     seq_dir[key] = seq_dir[key]['sequence']
    ###################################################################################################

    ###################################################################################################
    # protein_name = "DDX4"
    # seq_dir = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/DDX4/panel.csv", index_col=0, header=0, dtype=str).to_dict(orient='index')
    # # get the "mutant" column for key and "sequence" column for value
    # for key in seq_dir:
    #     seq_dir[key] = seq_dir[key]['sequence']
    ###################################################################################################

    ###################################################################################################
    # protein_name = "LAF-1_2"
    # seq_dir = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/LAF-1_2/panel.csv", index_col=0, header=0, dtype=str).to_dict(orient='index')
    # # get the "mutant" column for key and "sequence" column for value
    # for key in seq_dir:
    #     seq_dir[key] = seq_dir[key]['sequence']
    ###################################################################################################

    ###################################################################################################
    # seq_dir = {}
    # protein_name = "8991"
    # seq_dir[protein_name] = "PAYGMNGQMFHPMFFMPPQSMPPSMGMAGSMPGPPNMGNAMNGGPMVGSNSMMAPMMPAFMGQGMY"
    ###################################################################################################

    ###################################################################################################
    # seq_dir = {}
    # protein_name = "452"
    # seq_dir[protein_name] = "ANTFSNSASGGLFGQNNQQQGSGLFGQNSQTSGSSGLFGQNNQKQPNTFTQSNTGIGLFGQNNNQQ"
    ###################################################################################################

    ###################################################################################################
    # seq_dir = {}
    # protein_name = "13600"
    # seq_dir[protein_name] = "TSGSYGSSSKSSSYGKPKSGSYSKKPSYGGKKKSYGKKKSYNPPKGYGKKNKYNSSSGGGGGGGGG"
    ###################################################################################################

    ###################################################################################################
    # seq_dir = {}
    # protein_name = "13644"
    # seq_dir[protein_name] = "TSGSYGSSSRSSSYGRPRSGSYSRRPSYGGRRRSYGRRRSYNPPRGYGRRNRYNSSSGGGGGGGGG"
    ###################################################################################################

    ###################################################################################################
    # seq_dir = {}
    # protein_name = "14584"
    # seq_dir[protein_name] = "YSQGSGYNGYGGGGFNGWGGGSGFGRSGGYDGSNYGGGFGNYDGNYGGFGFGSPSGYGGQFSSGYN"
    ###################################################################################################

    ###################################################################################################
    # seq_dir = {}
    # protein_name = "13119"
    # seq_dir[protein_name] = "SSFSTFGNNSNFGGGTGFQGTTTFGSTTGGFLGLSFAGKPGTFGSGGKPFTGSGTNSGFTQASTGG"
    ###################################################################################################

   ###################################################################################################
    protein_name = "KH"
    seq_dir = {
        "KH-1": "MSPQAQQMNMNHNTMPSQFRDILRRQQMMQQQQQQGAGPGIGPGMANHNQFQQPQGVGYPPQQQQRMQHH", 
        "KH-2": "EAFFAAPNSISPLQSTSNSEQQAAFQQQAPISHIQTPMLSQEQAQPPQQGLFQPQVALGSLPPNPMPQSQ",
        "KH-3": "FTNANSYSTTTTTSNMGIMNFTTSGSSGTNSQGQTPQRVSGLQGSDALNIQQNQTSGGSLQAGQQKEGEQ",
        "KH-4": "SASSSERTIEESQTPAATESEAQSSSQLQPNGMQNAQDQSNSLQQVQIVGQPILQQIQIQQPQQQIIQAI",
        "KH-5": "VQIVQAQPQGQAQQAQSGTGQTMQVMQQIITNTGEIQQIPVQLNAGQLQYIRLAQPVSGTQVVQGQIQTL",
        "KH-6": "HLQQRPSGYVHQQAPTYGHGLTSTQRFSHQTLQQTPMISTMTPMSAQGVQAGVRSTAILPEQQQQQQQQQ",
        "KH-7": "NPELQPRTPRPASQSDAMDPLLSGLHIQQQSHPSGSLAPPHHPMQPVSVNRQMNPANFPQLQQQQQQQQQ",
    }
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

    scale = 'linear'  # 'linear' or 'log'
    min_protein_intensity_level = 20 if scale == 'log' else 0
    max_protein_intensity_level = 2048
    num_protein_intensity_levels = 4096
    batch_size = 256

    if scale == 'log':
        protein_intensity_levels = torch.logspace(
            start=np.log10(min_protein_intensity_level),
            end=np.log10(max_protein_intensity_level),
            steps=num_protein_intensity_levels,
        ).to(device)
    else:
        protein_intensity_levels = torch.linspace(
            min_protein_intensity_level, 
            max_protein_intensity_level, 
            steps=num_protein_intensity_levels
        ).to(device)

    for key, protein_seq in tqdm(seq_dir.items()):
        output_folder_name = f'protein_{key}'

        output_dir = Path(config.output_dir)
        output_dir = output_dir / protein_name / scale / output_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(protein_seq)

        protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)

        generated_imgs = []

        for batch_start in range(0, num_protein_intensity_levels, batch_size):
            batch_end = min(batch_start + batch_size, num_protein_intensity_levels)
            current_batch_size = batch_end - batch_start

            protein_intensity_level_batch = protein_intensity_levels[batch_start:batch_end]
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

        # save protein_intensity_levels as npy
        np.save(
            output_dir / "protein_intensity_levels.npy",
            protein_intensity_levels.cpu().numpy(),
        )

    logger.info("Done!")

if __name__ == "__main__":
    main()