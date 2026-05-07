# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellCropAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli
from esm.utils import encoding, decoding

from torchvision.utils import save_image

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
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = CELLFMConfig(**vars(args))

    dataset = OpenCellCropAllImageDataset(args, split_key='all')
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    vocab = dataset.vocab

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    gene_name = 'SAE1'
    index = 1

    dataset.meta_data = dataset.meta_data[dataset.meta_data['gene_name'] == gene_name]
    chosen_data = dataset.__getitem__(0)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_protein_img = chosen_data['protein_imgs'][index].unsqueeze(0).to(device)

    chosen_cell_img = chosen_nucleus_img

    num_val = 10

    # NLS
    test_signals = [
        'PKKKRKV',
        'KRPAATKKAGQAKKKK', 
        'AVKRPAATKKAGQAKKKKLD', 
        'MSRRRKANPTKLSENAKKLAKEVEN', 
        'PAAKRVKLD', 
        'KLKIKRPVK', 
        'PEFIRVKRRRDEDSVQALLIDEGKRVKKQKFIFK', 
        'RQQRKR', 
        'RRAMKRNARLRCPFRKGACEITRKTRR', 
        'PLRKAKR', 
        'KRRR', 
        'PAKRPR', 
        'PPPKDKKKKD', 
        'YGDGPRPPKMARYDNGSGY', 
        'RRRRHRNR', 
        'PRRVRLK', 
        'RRVPQRKEVSRCRKCRK', 
        'KRKR', 
        'RKKEAPGPREELRSRGR', 
        'PKKTQRR', 
        'KKRK', 
        'PYFRKRM', 
        'RRLGPTGKEVHALKRLRDS', 
    ]

    # NES
    # test_signals = [
    #     'EMFRELNEALEL', 
    #     'SLSFDESLALCVI', 
    #     'DKERWEDVKEEMTSALATMRVDYE', 
    #     'MTSALATMRV', 
    #     'LESNLRELQIC', 
    #     'LQLPPLERLTLD', 
    #     'ESFDIDDLCSKLKNKAKCS', 
    #     'IDMLIDLGLDLSD', 
    #     'LKEVDQLRLERLQIDEQLRQ', 
    #     'VTKRVVSLEKDTLLIDLHGTTQ', 
    #     'LIRTLKELKV', 
    #     'LTEQIHRLLM', 
    #     'LQLNLLQL', 
    #     'LDKLSVLRL', 
    #     'ILLRMSKMQL', 
    #     'HKLKITENSF', 
    #     'LLASFTSLSLQ', 
    # ]

    for i, test_signal in enumerate(test_signals):
        save_file = output_dir / '{:04d}'.format(i)
        save_file.mkdir(parents=True, exist_ok=True)

        # GFP
        seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
        # BLVRA
        # seq = 'MNAEPERKFGVVVVGVGRAGSVRMRDLRNPHPSSAFLNLIGFVSRRELGSIDGVQQISLEDALSSQEVEVAYICSESSSHEDYIRQFLNAGKHVLVEYPMTLSLAAAQELWELAEQKGKVLHEEHVELLMEEFAFLKKEVVGKDLLKGSLLFTAGPLEEERFGFPAFSGISRLTWLVSLFGELSLVSATLEERKEDQYMKMTVCLETEKKSPLSWIEEKGPGLKRNRYLSFHFKSGSLENVPNVGVNKNIFLKDQNIFVQKLLGQFSEKELAAEKKRILHCLGLAEEIQKYCCSRK'
        # MAPK9
        # seq = 'MSDSKCDSQFYSVQVADSTFTVLKRYQQLKPIGSGAQGIVCAAFDTVLGINVAVKKLSRPFQNQTHAKRAYRELVLLKCVNHKNIISLLNVFTPQKTLEEFQDVYLVMELMDANLCQVIHMELDHERMSYLLYQMLCGIKHLHSAGIIHRDLKPSNIVVKSDCTLKILDFGLARTACTNFMMTPYVVTRYYRAPEVILGMGYKENVDIWSVGCIMGELVKGCVIFQGTDHIDQWNKVIEQLGTPSAEFMKKLQPTVRNYVENRPKYPGIKFEELFPDWIFPSESERDKIKTSQARDLLSKMLVIDPDKRISVDEALRHPYITVWYDPAEAEAPPPQIYDAQLEEREHAIEEWKELIYKEVMDWEERSKNGVVKDQPSDAAVSSNATPSQSSSINDISSMSTEQTLASDTDSSLDASTGPLEGCR'

        # seq = "M" + test_signal + seq[1:]
        # seq = seq + test_signal
        # seq = "M" + test_signal + test_signal
        seq = test_signal
        # seq = "M" + test_signal + seq[1:] + test_signal
        # seq = seq

        protein_seq = encoding.tokenize_sequence(seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)

        # print(decoding.decode_sequence(protein_seq.squeeze(), vocab))
        print(test_signal)

        for j in range(num_val):
            sample = model.sequence_to_image(
                protein_seq, 
                chosen_cell_img, 
            )
            pred_img = torch.cat([torch.full_like(chosen_nucleus_img, -1), sample, chosen_nucleus_img], dim=1)

            save_image(pred_img, save_file / f'generated_img_{j:02d}.png', normalize=True, value_range=(-1, 1))
            save_image(sample, save_file / f'generated_protein_img_{j:02d}.png', normalize=True, value_range=(-1, 1))    

if __name__ == "__main__":
    main()
