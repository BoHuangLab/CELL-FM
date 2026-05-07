# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from cell_fm.models.cell_fm.cell_fm_cs_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_cs_model import CELLFMCSModel
from cell_fm.utils.cli_utils import cli

import tifffile as tiff
import numpy as np

from tqdm import tqdm
from esm.utils import encoding, decoding
import csv


@cli(CELLFMConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = CELLFMConfig(**vars(args))

    model = CELLFMCSModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    vocab = EsmSequenceTokenizer()

    num_aa = 66

    # expression_levels = [200, 500, 800, 1100, 1400, 1700, 2000]
    expression_levels = [200, 500, 800, 1100, 1400, 1700, 2000]
    num_gen = 100

    # Image without condensate
    # chosen_data_root = 'output/condenseq/seq2img_single/test/img_wo_condensate.tif'

    # Image with condensate
    chosen_data_root = 'output/condenseq/seq2img_single/test/img_with_condensate.tif'

    data = tiff.imread(chosen_data_root).astype(np.float32)
    chosen_cell_img = torch.from_numpy(data[0]).unsqueeze(0).unsqueeze(0).to(device)
    chosen_protein_img = torch.from_numpy(data[1]).unsqueeze(0).unsqueeze(0).to(device)

    # rescale to [0, 1]
    chosen_cell_img = chosen_cell_img / 65535.0
    chosen_protein_img = chosen_protein_img / 65535.0

    # normalize to [-1, 1]
    chosen_cell_img = chosen_cell_img * 2 - 1
    chosen_protein_img = chosen_protein_img * 2 - 1

    for expression_level in tqdm(expression_levels):
        from cell_fm.pipeline.accelerator.trainer import seed_everything
        seed_everything(6)

        gen_signals = []

        masked_seq = "<mask>" * num_aa

        config.output_dir = f'output/condenseq/img2seq/el_{expression_level:04d}'
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        expression_level = torch.tensor([expression_level], dtype=torch.float32, device=device).unsqueeze(0)

        protein_seq = encoding.tokenize_sequence(masked_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)
        protein_seq_mask = (protein_seq == vocab.mask_token_id)

        for i in tqdm(range(num_gen)):
            sample = model.image_to_sequence(
                protein_seq, protein_seq_mask, chosen_protein_img, 
                chosen_cell_img, expression_level, order='l2r', temperature=2.0, 
                progress=False, 
            )
            gen_seq = decoding.decode_sequence(sample.squeeze(), vocab)
            gen_sig = gen_seq

            print(gen_sig)

            if gen_sig not in gen_signals:
                gen_signals.append(gen_sig)

        # save gen_signals
        with open(output_dir / 'gen_signals.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for signal in gen_signals:
                writer.writerow([signal])

if __name__ == "__main__":
    main()