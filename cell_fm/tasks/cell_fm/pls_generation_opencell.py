# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path
import torch

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellCropAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli

from torchvision.utils import save_image
from esm.utils import encoding, decoding
import csv


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


@cli(CELLFMConfig)
def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))

    dataset = OpenCellCropAllImageDataset(args, split_key='all')
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    vocab = dataset.vocab

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    # Nucleoli
    # gene_name = 'AATF'
    # index = 1

    # NLS
    gene_name = 'SAE1'
    index = 1

    # NES
    # gene_name = 'RIOK2'
    # index = 21

    dataset.meta_data = dataset.meta_data[dataset.meta_data['gene_name'] == gene_name]
    chosen_data = dataset.__getitem__(0)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_protein_img = chosen_data['protein_imgs'][index].unsqueeze(0).to(device)

    chosen_cell_img = chosen_nucleus_img

    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(chosen_nucleus_img, output_dir / 'chosen_nucleus_img.png', normalize=True, value_range=(-1, 1))
    save_image(chosen_protein_img, output_dir / 'chosen_protein_img.png', normalize=True, value_range=(-1, 1))

    cat_img = torch.cat([torch.full_like(chosen_nucleus_img, -1), chosen_protein_img, chosen_nucleus_img], dim=1)
    save_image(cat_img, output_dir / 'cat_img.png', normalize=True, value_range=(-1, 1))

    # num_aas = [105]
    num_aas = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    num_gen = 40
    num_val = 1
    gen_pos = "end"

    gen_signals = []

    # GFP
    seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    # BLVRA
    # seq = 'MNAEPERKFGVVVVGVGRAGSVRMRDLRNPHPSSAFLNLIGFVSRRELGSIDGVQQISLEDALSSQEVEVAYICSESSSHEDYIRQFLNAGKHVLVEYPMTLSLAAAQELWELAEQKGKVLHEEHVELLMEEFAFLKKEVVGKDLLKGSLLFTAGPLEEERFGFPAFSGISRLTWLVSLFGELSLVSATLEERKEDQYMKMTVCLETEKKSPLSWIEEKGPGLKRNRYLSFHFKSGSLENVPNVGVNKNIFLKDQNIFVQKLLGQFSEKELAAEKKRILHCLGLAEEIQKYCCSRK'
    # MAPK9
    # seq = 'MSDSKCDSQFYSVQVADSTFTVLKRYQQLKPIGSGAQGIVCAAFDTVLGINVAVKKLSRPFQNQTHAKRAYRELVLLKCVNHKNIISLLNVFTPQKTLEEFQDVYLVMELMDANLCQVIHMELDHERMSYLLYQMLCGIKHLHSAGIIHRDLKPSNIVVKSDCTLKILDFGLARTACTNFMMTPYVVTRYYRAPEVILGMGYKENVDIWSVGCIMGELVKGCVIFQGTDHIDQWNKVIEQLGTPSAEFMKKLQPTVRNYVENRPKYPGIKFEELFPDWIFPSESERDKIKTSQARDLLSKMLVIDPDKRISVDEALRHPYITVWYDPAEAEAPPPQIYDAQLEEREHAIEEWKELIYKEVMDWEERSKNGVVKDQPSDAAVSSNATPSQSSSINDISSMSTEQTLASDTDSSLDASTGPLEGCR'

    for num_aa in num_aas:
        if gen_pos == 'start':
            masked_seq = "M" + "<mask>" * num_aa + seq[1:]
        elif gen_pos == 'end':
            masked_seq = seq + "<mask>" * num_aa
            # masked_seq = "M" + "<mask>" * num_aa

        print(masked_seq)

        protein_seq = encoding.tokenize_sequence(masked_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)
        protein_seq_mask = (protein_seq == vocab.mask_token_id)

        for i in range(num_gen):
            sample = model.image_to_sequence(
                protein_seq, protein_seq_mask, chosen_protein_img, 
                chosen_cell_img, order='random', temperature=1.0, progress=False, 
            )
            gen_seq = decoding.decode_sequence(sample.squeeze(), vocab)

            if gen_pos == 'start':
                gen_sig = gen_seq[1:num_aa+1]
            elif gen_pos == 'end':
                gen_sig = gen_seq[-num_aa:]

            if gen_sig not in gen_signals:
                gen_signals.append(gen_sig)

                seq_save = [gen_seq, gen_sig]
                save_dir = output_dir / '{:04d}_{:02d}'.format(num_aa, i+1)
                save_dir.mkdir(parents=True, exist_ok=True)

                with open(save_dir / 'seq.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    for item in seq_save:
                        writer.writerow([item])

                for j in range(num_val):
                    # test_seq = seq + gen_sig * 2
                    # test_seq = encoding.tokenize_sequence(test_seq, vocab, True)
                    # test_seq = test_seq.unsqueeze(0).to(device)
                    # protein_image_gen = model.sequence_to_image(
                    #     test_seq, chosen_cell_img, config.num_steps, 
                    # )

                    protein_image_gen = model.sequence_to_image(
                        sample, chosen_cell_img, config.num_steps, 
                    )

                    save_image(
                        protein_image_gen, 
                        save_dir / 'protein_image_gen_{:02d}.png'.format(j+1), 
                        normalize=True, 
                        value_range=(-1, 1)
                    )

    # save gen_signals
    with open(output_dir / 'gen_signals.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for signal in gen_signals:
            writer.writerow([signal])

if __name__ == "__main__":
    main()