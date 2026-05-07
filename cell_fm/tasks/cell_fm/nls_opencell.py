# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.unidiff import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellLMDBDataset
from cell_fm.models.cell_fm_v4.config import CELLDiffConfig
from cell_fm.models.cell_fm_v4.model import CELLDiffModel
from cell_fm.models.vae.vae_model import VAEModel
from cell_fm.models.vae.vae_config import VAEConfig
from cell_fm.utils.cli_utils import cli
from cell_fm.data.hpa_data.vocabulary import convert_string_sequence_to_int_index
import pandas as pd
import numpy as np

from torchvision.utils import save_image
from copy import deepcopy

def colorize_image(tensor, color):
    # Create a zero tensor with the same size as the input tensor but with three channels
    colored_image = torch.zeros((3, tensor.size(0), tensor.size(1)), dtype=tensor.dtype, device=tensor.device)

    if color == 'blue':
        colored_image[2] = tensor  # Set blue channel
    elif color == 'red':
        colored_image[0] = tensor  # Set red channel
    elif color == 'green':
        colored_image[1] = tensor  # Set green channel
    elif color == 'yellow':
        colored_image[0] = tensor  # Set red and green channels to get yellow
        colored_image[1] = tensor
    return colored_image

def save_colored_image(tensor, filename, color):
    colored_tensor = colorize_image(tensor, color)
    save_image(colored_tensor, filename, normalize=True, value_range=(0, 1))


@cli(CELLDiffConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    vae_args = deepcopy(args)
    vae_args.infer = True

    vae = VAEModel(config=VAEConfig(**vars(vae_args)))

    for param in vae.parameters():
        param.requires_grad = False
    vae.to(device)
    vae.eval()

    valset = OpenCellLMDBDataset(args, split_key=args.split_key, vae=vae)
    vocab = valset.vocab

    model = CELLDiffModel(config=CELLDiffConfig(**vars(args)), loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.split_key

    with torch.no_grad():
        chosen_data = valset.__getitem__(0)
        chosen_nucleus_img = chosen_data['nucleus_img'].unsqueeze(0).to(device)

        chosen_nucleus_img_latent = vae.encode(chosen_nucleus_img).sample()
        cell_img_latent = chosen_nucleus_img_latent
    
    file_path = 'yeast.csv'
    df = pd.read_csv(file_path)
    nls_list = df['NLS'].tolist()

    from PIL import Image

    nucl_mask = Image.open('output/opencell/pls_generation/PT_CELL-Diff-LD_CELL-E2-split/nes_masks.png').convert('L')
    nucl_mask = (np.array(nucl_mask) > 1)

    cell_mask = ~nucl_mask

    cell_mask = cell_mask | nucl_mask
    cp_mask = cell_mask ^ nucl_mask

    counts = []

    for i, nls in enumerate(nls_list):
        save_file = output_dir / '{:04d}'.format(i)
        save_file.mkdir(parents=True, exist_ok=True)

        # GFP
        seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
        # BLVRA
        # seq = 'MNAEPERKFGVVVVGVGRAGSVRMRDLRNPHPSSAFLNLIGFVSRRELGSIDGVQQISLEDALSSQEVEVAYICSESSSHEDYIRQFLNAGKHVLVEYPMTLSLAAAQELWELAEQKGKVLHEEHVELLMEEFAFLKKEVVGKDLLKGSLLFTAGPLEEERFGFPAFSGISRLTWLVSLFGELSLVSATLEERKEDQYMKMTVCLETEKKSPLSWIEEKGPGLKRNRYLSFHFKSGSLENVPNVGVNKNIFLKDQNIFVQKLLGQFSEKELAAEKKRILHCLGLAEEIQKYCCSRK'

        # seq = "M" + nls + seq[1:]
        # seq = seq + nls
        seq = seq + nls + nls
        # seq = "M" + nls + nls
        # seq = seq
        # seq = nls
        # seq = "M" + nls + seq[1:] + nls
        protein_seq = convert_string_sequence_to_int_index(vocab, seq)
        protein_seq = torch.LongTensor(protein_seq).unsqueeze(0).to(device)

        real_protein_seq = vocab.untokenize(protein_seq.squeeze()[1:-1])
        print(nls)
        
        sample = model.sequence_to_image(
            protein_seq, 
            cell_img_latent, 
            sampling_strategy="ddim", 
            progress=False
        )
        sample = vae.decode(sample).sample
        pred_img = torch.cat([torch.full_like(chosen_nucleus_img, -1), sample, chosen_nucleus_img], dim=1)

        save_image(pred_img, save_file / 'generated_img.png', normalize=True, value_range=(-1, 1))
        save_image(sample, save_file / 'generated_protein_img.png', normalize=True, value_range=(-1, 1))    
        save_image(3 * (sample + 1) / 2, save_file / 'generated_protein_img_amp_s3.png', normalize=True, value_range=(0, 1))   
        save_image(5 * (sample + 1) / 2, save_file / 'generated_protein_img_amp_s5.png', normalize=True, value_range=(0, 1))    
        save_image(8 * (sample + 1) / 2, save_file / 'generated_protein_img_amp_s8.png', normalize=True, value_range=(0, 1))   

        sample_np = sample.detach().cpu().squeeze().numpy()

        in_intensity_median = np.median(sample_np[nucl_mask])
        out_intensity_median = np.median(sample_np[cp_mask])

        count = 1 if (in_intensity_median > out_intensity_median) else 0
        print(count)
        counts.append(count)

    print(sum(counts) / len(counts))

if __name__ == "__main__":
    main()
