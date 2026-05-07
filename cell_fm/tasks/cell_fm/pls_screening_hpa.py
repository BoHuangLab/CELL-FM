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

    with torch.no_grad():
        chosen_data = valset.__getitem__(3)
        chosen_nucleus_img = chosen_data['nucleus_img'].unsqueeze(0).to(device)
        chosen_ER_img = chosen_data['ER_img'].unsqueeze(0).to(device)
        chosen_microtubules_img = chosen_data['microtubules_img'].unsqueeze(0).to(device)

        chosen_nucleus_img_latent = vae.encode(chosen_nucleus_img).sample()
        chosen_microtubules_img_latent = vae.encode(chosen_microtubules_img).sample()
        chosen_ER_img_latent = vae.encode(chosen_ER_img).sample()

        if args.cell_image == 'nucl':
            cell_img_latent = chosen_nucleus_img_latent
        elif args.cell_image == 'nucl,er':
            if args.test_cell_image == 'nucl':
                chosen_ER_img_latent = model.cell_img_placeholder.expand(
                    chosen_nucleus_img_latent.shape[0], 
                    chosen_nucleus_img_latent.shape[1], 
                    *model.cell_img_placeholder.shape
                )
            cell_img_latent = torch.cat([chosen_nucleus_img_latent, chosen_ER_img_latent], dim=1)
        elif args.cell_image == 'nucl,mt':
            if args.test_cell_image == 'nucl':
                chosen_microtubules_img_latent = model.cell_img_placeholder.expand(
                    chosen_nucleus_img_latent.shape[0], 
                    chosen_nucleus_img_latent.shape[1], 
                    *model.cell_img_placeholder.shape
                )
            cell_img_latent = torch.cat([chosen_nucleus_img_latent, chosen_microtubules_img_latent], dim=1)
        elif args.cell_image == 'nucl,er,mt':
            if args.test_cell_image == 'nucl':
                chosen_ER_img_latent = model.cell_img_placeholder.expand(
                    chosen_nucleus_img_latent.shape[0], 
                    chosen_nucleus_img_latent.shape[1], 
                    *model.cell_img_placeholder.shape
                )
                chosen_microtubules_img_latent = model.cell_img_placeholder.expand(
                    chosen_nucleus_img_latent.shape[0], 
                    chosen_nucleus_img_latent.shape[1], 
                    *model.cell_img_placeholder.shape
                )
            elif args.test_cell_image == 'nucl,er':
                chosen_microtubules_img_latent = model.cell_img_placeholder.expand(
                    chosen_nucleus_img_latent.shape[0], 
                    chosen_nucleus_img_latent.shape[1], 
                    *model.cell_img_placeholder.shape
                )
            elif args.test_cell_image == 'nucl,mt':
                chosen_ER_img_latent = model.cell_img_placeholder.expand(
                    chosen_nucleus_img_latent.shape[0], 
                    chosen_nucleus_img_latent.shape[1], 
                    *model.cell_img_placeholder.shape
                )
            cell_img_latent = torch.cat([chosen_nucleus_img_latent, chosen_ER_img_latent, chosen_microtubules_img_latent], dim=1)
        else:
            raise ValueError(f"Cell image type: {args.cell_image} is not supported")

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
        'NQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGY', 
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
    ]

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
    #     'HKLKITENSF'
    # ]

    for i, test_signal in enumerate(test_signals):
        save_file = output_dir / '{:04d}'.format(i)
        save_file.mkdir(parents=True, exist_ok=True)

        # GFP
        # seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
        # BLVRA
        seq = 'MNAEPERKFGVVVVGVGRAGSVRMRDLRNPHPSSAFLNLIGFVSRRELGSIDGVQQISLEDALSSQEVEVAYICSESSSHEDYIRQFLNAGKHVLVEYPMTLSLAAAQELWELAEQKGKVLHEEHVELLMEEFAFLKKEVVGKDLLKGSLLFTAGPLEEERFGFPAFSGISRLTWLVSLFGELSLVSATLEERKEDQYMKMTVCLETEKKSPLSWIEEKGPGLKRNRYLSFHFKSGSLENVPNVGVNKNIFLKDQNIFVQKLLGQFSEKELAAEKKRILHCLGLAEEIQKYCCSRK'
        # MAPK9
        # seq = 'MSDSKCDSQFYSVQVADSTFTVLKRYQQLKPIGSGAQGIVCAAFDTVLGINVAVKKLSRPFQNQTHAKRAYRELVLLKCVNHKNIISLLNVFTPQKTLEEFQDVYLVMELMDANLCQVIHMELDHERMSYLLYQMLCGIKHLHSAGIIHRDLKPSNIVVKSDCTLKILDFGLARTACTNFMMTPYVVTRYYRAPEVILGMGYKENVDIWSVGCIMGELVKGCVIFQGTDHIDQWNKVIEQLGTPSAEFMKKLQPTVRNYVENRPKYPGIKFEELFPDWIFPSESERDKIKTSQARDLLSKMLVIDPDKRISVDEALRHPYITVWYDPAEAEAPPPQIYDAQLEEREHAIEEWKELIYKEVMDWEERSKNGVVKDQPSDAAVSSNATPSQSSSINDISSMSTEQTLASDTDSSLDASTGPLEGCR'

        # test_signal = "MREIVHLQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLERINVYYNEATGGKYVPRAVLVDLEPGTMDSVRSGPFGQIFRPDNFVFGQSGAGNNWAKGHYTEGAELVDSVLDVVRKEAESCDCLQGFQLTHSLGGGTGSGMGTLLISKIREEYPDRIMNTFSVVPSPKVSDTVVEPYNATLSVHQLVENTDETYCIDNEALYDICFRTLKLTTPTYGDLNHLVSATMSGVTTCLRFPGQLNADLRKLAVNMVPFPRLHFFMPGFAPLTSRGSQQYRALTVPELTQQMFDAKNMMAACDPRHGRYLTVAAVFRGRMSMKEVDEQMLNVQNKNSSYFVEWIPNNVKTAVCDIPPRGLKMSATFIGNSTAIQELFKRISEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDATAEEEGEFEEEAEEEVA"

        # seq = "M" + test_signal + seq
        # seq = seq + test_signal
        # seq = test_signal
        # seq = seq
        seq = "M" + test_signal + seq[1:] + test_signal
        protein_seq = convert_string_sequence_to_int_index(vocab, seq)
        protein_seq = torch.LongTensor(protein_seq).unsqueeze(0).to(device)
        
        real_protein_seq = vocab.untokenize(protein_seq.squeeze())
        print(real_protein_seq)
        
        sample = model.sequence_to_image(protein_seq, cell_img_latent, sampling_strategy="ddim")
        pred_img = torch.cat([torch.full_like(chosen_nucleus_img, -1), sample, chosen_nucleus_img], dim=1)

        save_image(pred_img, save_file / 'generated_img.png', normalize=True, value_range=(-1, 1))
        save_image(sample, save_file / 'generated_protein_img.png', normalize=True, value_range=(-1, 1))    
        save_image(3 * (sample + 1) / 2, save_file / 'generated_protein_img_amp.png', normalize=True, value_range=(0, 1))    

if __name__ == "__main__":
    main()
