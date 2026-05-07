# -*- coding: utf-8 -*-
import os
import sys

import torch
import numpy as np

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

from cell_fm.data.hpa_data.vocabulary import convert_string_sequence_to_int_index

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # valset = HPALMDBDataset(args, split_key='virtual_staining')
    # chosen_data = valset.__getitem__(16)

    # Nucleoli 
    # chosen_data = valset.__getitem__(489)
    
    # NLS
    chosen_data = valset.__getitem__(720)
    # chosen_data = valset.__getitem__(217)

    # NES
    # chosen_data = valset.__getitem__(506)

    chosen_nucleus_img = chosen_data['nucleus_img'].unsqueeze(0).to(device)
    chosen_ER_img = chosen_data['ER_img'].unsqueeze(0).to(device)
    chosen_microtubules_img = chosen_data['microtubules_img'].unsqueeze(0).to(device)
    chosen_protein_img = chosen_data['protein_img'].unsqueeze(0).to(device)

    protein_img_latent = vae.encode(chosen_protein_img).sample()
    nucleus_img_latent = vae.encode(chosen_nucleus_img).sample()
    microtubules_img_latent = vae.encode(chosen_microtubules_img).sample()
    ER_img_latent = vae.encode(chosen_ER_img).sample()

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

    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(chosen_nucleus_img, output_dir / 'chosen_nucleus_img.png', normalize=True, value_range=(-1, 1))
    save_image(chosen_protein_img, output_dir / 'chosen_protein_img.png', normalize=True, value_range=(-1, 1))
    save_image(chosen_ER_img, output_dir / 'chosen_ER_img.png', normalize=True, value_range=(-1, 1))
    save_image(chosen_microtubules_img, output_dir / 'chosen_microtubules_img.png', normalize=True, value_range=(-1, 1))

    cat_img = torch.cat([torch.full_like(chosen_nucleus_img, -1), chosen_protein_img, chosen_nucleus_img], dim=1)
    save_image(cat_img, output_dir / 'cat_img.png', normalize=True, value_range=(-1, 1))

    # seq = 'MAAYKKKKPNHKRRYGDSDSRGKKKSKKKTVERGSKKDYDQAGPSAGRRDHASKEDDSQSDLAELLRISSTGTKSEDGPPEAKKSKKRKKKPRNPTSSDFFCKKKYKDLKEEHQNRYFSLQVKTANLLEQNNLLNDEQCYFFTTQVEEKLSKMEEDVAEYKSDLKELRRKLALPEVASKALEGVAAVGMVSGGERSKSV'
    # seq = 'MAPAAATLANAEAKRGLIDVRKSLRKVRITNEGESQKERDSQGPKDEARKGEFWKDRNGGKAGKFQRGRALTEFPVVHEAIADDEKIPWPERIPLTKSSLLTLTCERRSMAGSKMRKIFGPDVKAELGSKWNVLTTQAFIKFTGAWSELEEREEKHKAYKKKSQKERKAQDALGKVAKKQQQALEDKLGFERRELKVEL'
    # seq = 'MAIVDPKAKKNKEKKGAFFVQVKRAEKGLKVEKVPKKAKKQKKNGGEKQKTQAGVDGRAFVEVETEEKRIDVEAKGDVGPADGGKKKKKPEGPVRPATAFFLFCSEFRPKIKSTNRGISIPDVNAKLGEMWNNLTDEDKQPFITKAAELKEEYEKDKADKEKSKKDGKAKHALKVGRKKVEEEDEDEDEEEEGEEEEEY'
    # seq = 'MREIVHLQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLERINVYYNEATGGKYVPRAVLVDLEPGTMDSVRSGPFGQIFRPDNFVFGQSGAGNNWAKGHYTEGAELVDSVLDVVRKEAESCDCLQGFQLTHSLGGGTGSGMGTLLISKIREEYPDRIMNTFSVVPSPKVSDTVVEPYNATLSVHQLVENTDETYCIDNEALYDICFRTLKLTTPTYGDLNHLVSATMSGVTTCLRFPGQLNADLRKLAVNMVPFPRLHFFMPGFAPLTSRGSQQYRALTVPELTQQMFDAKNMMAACDPRHGRYLTVAAVFRGRMSMKEVDEQMLNVQNKNSSYFVEWIPNNVKTAVCDIPPRGLKMSATFIGNSTAIQELFKRISEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDATAEEEGEFEEEAEEEVA'
    # seq = 'MREIVHLQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLERINVYYNEATGGKYVPRAVLVDLEPGTMDSVRSGPFGQIFRPDNFVFGQSGAGNNWAKGHYTEGAELVDSVLDVVRKEAESCDCLQGFQLTHSLGGGTGSGMGTLLISKIREEYPDRIMNTFSVVPSPKVSDTVVEPYNATLSVHQLVENTDETYCIDNEALYDICFRTLKLTTPTYGDLNHLVSATMSGVTTCLRFPGQLNADLRKLAVNMVPFPRLHFFMPGFAPLTSRGSNMMAACDPRHGRYLTVAAVFRGRMSMKEVDEQMLNVQNKNSSYFVEWIPNNVKTAVCDIPPRGLKMSATFIGNSTAIQELFKRISEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDATG'
    # seq = 'MREIVHLQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLERINVYYNEATGGKYVPRAVLVDLEP'
    # GFP
    # seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    # 
    # seq = 'MPRQGSLGAAPPKVAPDSSETVVGQKRPKGKSPRKRPKAAGKRQRKRRGGY'
    seq = 'MPRQGSLGAAPPKVAPDSSETVVG'

    protein_seq = convert_string_sequence_to_int_index(vocab, seq)
    protein_seq = torch.LongTensor(protein_seq).unsqueeze(0).to(device)

    print(vocab.untokenize(protein_seq.squeeze()[1:-1]))

    num_gen = 10

    for i in range(num_gen):

        protein_image_gen = model.sequence_to_image(
            protein_seq, cell_img_latent, sampling_strategy="ddim", progress=False, get_attention_map=True
        )
        attn_maps = model.attn_maps
        attn_map = torch.norm(torch.cat(attn_maps, dim=0), dim=0, keepdim=True)

        import matplotlib.pyplot as plt
        def plot_attention(attn_map, head=0, root=None):
            attn = attn_map[0, head].detach().cpu().numpy()
            plt.imshow(attn, cmap="viridis")
            plt.colorbar()
            plt.title(f"Attention Head {head}")
            plt.savefig(root)

        plot_attention(attn_map, head=0, root = output_dir / f'attn_map_{i}.png')

        attn_map = attn_map.norm(dim=1).squeeze()
        attn_seq_to_image = attn_map[:64, 65:-1]

        plt.figure(figsize=(10, 6))
        plt.imshow(attn_seq_to_image.cpu().numpy(), cmap="magma", aspect="auto")
        plt.colorbar()
        plt.xlabel("Image Tokens")
        plt.ylabel("Sequence Tokens")
        plt.title("Sequence Tokens Attending to Image Tokens")
        plt.savefig(output_dir / f'attn_seq_to_img_{i}.png')

        protein_image_gen = vae.decode(protein_image_gen).sample
        
        seq_weight = attn_seq_to_image.sum(dim=0)

        seq_weight = seq_weight.cpu().numpy()
        seq_weight = seq_weight - seq_weight.min()

        x_positions = np.arange(len(seq))
        plt.figure(figsize=(max(10, len(seq) * 0.2), 6))

        plt.bar(x_positions, seq_weight, color="purple", alpha=0.7)

        plt.xticks(x_positions, list(seq), rotation=0, fontsize=8)

        plt.xlabel("Amino Acid Position")
        plt.ylabel("Attention Weight")
        plt.title("Sequence Token Attention Weights")

        plt.savefig(output_dir / f"seq_attention_weights_{i}.png", bbox_inches="tight", dpi=300)

        save_image(
            protein_image_gen, 
            output_dir / f'protein_image_gen_{i}.png', 
            normalize=True, 
            value_range=(-1, 1)
        )

    
if __name__ == "__main__":
    main()
