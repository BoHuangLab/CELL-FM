# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.unidiff import UniDiffCriterions
from cell_fm.data.hpa_data.dataset import HPALMDBDataset
from cell_fm.models.cell_fm_v4.config import CELLDiffConfig
from cell_fm.models.cell_fm_v4.model import CELLDiffModel
from cell_fm.models.vae.vae_model import VAEModel
from cell_fm.models.vae.vae_config import VAEConfig
from cell_fm.utils.cli_utils import cli
from cell_fm.logging import logger
from cell_fm.data.hpa_data.vocabulary import convert_string_sequence_to_int_index

from cell_fm.metrics.iou import compute_iou, binarize_img
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

    valset = HPALMDBDataset(args, split_key=args.split_key, vae=vae)
    vocab = valset.vocab

    model = CELLDiffModel(config=CELLDiffConfig(**vars(args)), loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.split_key

    # MT
    # valset = HPALMDBDataset(args, split_key='virtual_staining', vae=vae)
    # chosen_data = valset.__getitem__(16)

    # Nucleoli 
    # chosen_data = valset.__getitem__(489)
    
    # NLS
    chosen_data = HPALMDBDataset(args, split_key='test', vae=vae).__getitem__(720)
    # chosen_data = valset.__getitem__(217)

    # NES
    # chosen_data = valset.__getitem__(506)
    # chosen_data = HPALMDBDataset(args, split_key='cell_e2_test', vae=vae).__getitem__(559)

    with torch.no_grad():
        protein_seq = chosen_data['protein_seq'].unsqueeze(0).to(device)
        protein_img = chosen_data['protein_img'].unsqueeze(0).to(device)
        nucleus_img = chosen_data['nucleus_img'].unsqueeze(0).to(device)
        microtubules_img = chosen_data['microtubules_img'].unsqueeze(0).to(device)
        ER_img = chosen_data['ER_img'].unsqueeze(0).to(device)            

        protein_img_latent = vae.encode(protein_img).sample()
        nucleus_img_latent = vae.encode(nucleus_img).sample()
        microtubules_img_latent = vae.encode(microtubules_img).sample()
        ER_img_latent = vae.encode(ER_img).sample()

    if args.cell_image == 'nucl':
        cell_img_latent = nucleus_img_latent
    elif args.cell_image == 'nucl,er':
        if args.test_cell_image == 'nucl':
            ER_img_latent = model.cell_img_placeholder.expand(
                protein_img_latent.shape[0], 
                protein_img_latent.shape[1], 
                *model.cell_img_placeholder.shape
            )
        cell_img_latent = torch.cat([nucleus_img_latent, ER_img_latent], dim=1)
    elif args.cell_image == 'nucl,mt':
        if args.test_cell_image == 'nucl':
            microtubules_img_latent = model.cell_img_placeholder.expand(
                protein_img_latent.shape[0], 
                protein_img_latent.shape[1], 
                *model.cell_img_placeholder.shape
            )
        cell_img_latent = torch.cat([nucleus_img_latent, microtubules_img_latent], dim=1)
    elif args.cell_image == 'nucl,er,mt':
        if args.test_cell_image == 'nucl':
            ER_img_latent = model.cell_img_placeholder.expand(
                protein_img_latent.shape[0], 
                protein_img_latent.shape[1], 
                *model.cell_img_placeholder.shape
            )
            microtubules_img_latent = model.cell_img_placeholder.expand(
                protein_img_latent.shape[0], 
                protein_img_latent.shape[1], 
                *model.cell_img_placeholder.shape
            )
        elif args.test_cell_image == 'nucl,er':
            microtubules_img_latent = model.cell_img_placeholder.expand(
                protein_img_latent.shape[0], 
                protein_img_latent.shape[1], 
                *model.cell_img_placeholder.shape
            )
        elif args.test_cell_image == 'nucl,mt':
            ER_img_latent = model.cell_img_placeholder.expand(
                protein_img_latent.shape[0], 
                protein_img_latent.shape[1], 
                *model.cell_img_placeholder.shape
            )
        cell_img_latent = torch.cat([nucleus_img_latent, ER_img_latent, microtubules_img_latent], dim=1)
    else:
        raise ValueError(f"Cell image type: {args.cell_image} is not supported")
    
    num_aas = [10, 20, 30, 40]
    num_seq_gen = 10

    seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

    for i, num_aa in enumerate(num_aas):
        ious = []
        for j in range(num_seq_gen):
            masked_seq = seq + "<mask>" * num_aa
            # masked_seq = "M" + "<mask>" * num_aa

            protein_seq = convert_string_sequence_to_int_index(vocab, masked_seq)
            protein_seq = torch.LongTensor(protein_seq).unsqueeze(0).to(device)

            # print(vocab.untokenize(protein_seq.squeeze()))
            protein_seq_mask = (protein_seq == vocab.mask_idx)

            seq_gen = model.image_to_sequece(
                protein_seq, protein_seq_mask, protein_img_latent, 
                cell_img_latent, order='random', temperature=1.0, 
                progress=False, 
            )

            img_gen = model.sequence_to_image(
                seq_gen, 
                cell_img_latent, 
                sampling_strategy="ddim", 
                progress=False
            )

            img_gen = vae.decode(img_gen).sample

            save_file = output_dir / ('num_aa_{:04d}_{:04d}'.format(num_aa, j+1))
            save_file.mkdir(parents=True, exist_ok=True)

            pred_img = torch.cat([torch.full_like(protein_img, -1), img_gen, nucleus_img], dim=1)
            save_image(pred_img, save_file / 'generated_img.png', normalize=True, value_range=(-1, 1))
            save_image(img_gen, save_file / 'generated_protein_img.png', normalize=True, value_range=(-1, 1))
            save_image(protein_img, save_file / 'real_protein_img.png', normalize=True, value_range=(-1, 1))

            iou = compute_iou(binarize_img(img_gen, threshold_mode="quantile", quantile_q=0.5), binarize_img(protein_img)).item()
            ious.append(iou)

            # generated_threshold_img = binarize_img(img_gen, threshold_mode="quantile", quantile_q=0.5)
            # real_threshold_img = binarize_img(protein_img)

            # real_threshold_img = 2 * (real_threshold_img.float() * 0.5) - 1
            # real_threshold_img = torch.cat([torch.full_like(protein_img, -1), real_threshold_img, nucleus_img], dim=1)
            # save_image(real_threshold_img, save_file / 'real_threshold_img.png', normalize=True, value_range=(-1, 1))        

            # generated_threshold_img = 2 * (generated_threshold_img.float() * 0.5) - 1
            # generated_threshold_img = torch.cat([torch.full_like(protein_img, -1), generated_threshold_img, nucleus_img], dim=1)
            # save_image(generated_threshold_img, save_file / 'generated_threshold_img.png', normalize=True, value_range=(-1, 1))

            # save_colored_image((nucleus_img.squeeze() + 1) / 2, save_file / 'real_nucleus_img_blue.png', 'blue')
            # save_colored_image((microtubules_img.squeeze() + 1) / 2, save_file / 'real_microtubules_img_red.png', 'red')
            # save_colored_image((protein_img.squeeze() + 1) / 2, save_file / 'real_protein_img_green.png', 'green')
            # save_colored_image((ER_img.squeeze() + 1) / 2, save_file / 'real_ER_img_yellow.png', 'yellow')
            # save_colored_image((img_gen.squeeze() + 1) / 2, save_file / 'generated_protein_img_green.png', 'green')

            # logger.info(f"iou: {iou}")
        logger.info(f"Num of aa: {num_aa}")
        logger.info(f"Avg iou: {sum(ious)/len(ious)}")

if __name__ == "__main__":
    main()