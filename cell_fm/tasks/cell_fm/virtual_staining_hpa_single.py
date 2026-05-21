# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.hpa_data.dataset import HPAAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli

from torchvision.utils import save_image
from esm.utils import encoding, decoding


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
    dataset = HPAAllImageDataset(config, split_key='all')
    vocab = dataset.vocab

    dataset.meta_data = dataset.meta_data[dataset.meta_data['gene_name'] == 'H3C13']
    chosen_data = dataset.__getitem__(0)
    index = 33
    # dataset.meta_data = dataset.meta_data[dataset.meta_data['gene_name'] == 'BLVRA']
    # chosen_data = dataset.__getitem__(0)
    # index = 6

    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)
    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_microtubules_img = chosen_data['microtubules_imgs'][index].unsqueeze(0).to(device)
    chosen_ER_img = chosen_data['ER_imgs'][index].unsqueeze(0).to(device)

    num_samples = 20
    batch_size = 32

    with torch.no_grad():
        if config.cell_image == 'nucl':
            chosen_cell_img = chosen_nucleus_img
        elif config.cell_image == 'nucl,er':
            if config.test_cell_image == 'nucl':
                chosen_ER_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            chosen_cell_img = torch.cat([chosen_nucleus_img, chosen_ER_img], dim=1)
        elif config.cell_image == 'nucl,mt':
            if config.test_cell_image == 'nucl':
                chosen_microtubules_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            chosen_cell_img = torch.cat([chosen_nucleus_img, chosen_microtubules_img], dim=1)
        elif config.cell_image == 'nucl,er,mt':
            if config.test_cell_image == 'nucl':
                chosen_ER_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
                chosen_microtubules_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            elif config.test_cell_image == 'nucl,er':
                chosen_microtubules_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            elif config.test_cell_image == 'nucl,mt':
                chosen_ER_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            chosen_cell_img = torch.cat([chosen_nucleus_img, chosen_ER_img, chosen_microtubules_img], dim=1)
        else:
            raise ValueError(f"Cell image type: {config.cell_image} is not supported")

    blvra = 'MNAEPERKFGVVVVGVGRAGSVRMRDLRNPHPSSAFLNLIGFVSRRELGSIDGVQQISLEDALSSQEVEVAYICSESSSHEDYIRQFLNAGKHVLVEYPMTLSLAAAQELWELAEQKGKVLHEEHVELLMEEFAFLKKEVVGKDLLKGSLLFTAGPLEEERFGFPAFSGISRLTWLVSLFGELSLVSATLEERKEDQYMKMTVCLETEKKSPLSWIEEKGPGLKRNRYLSFHFKSGSLENVPNVGVNKNIFLKDQNIFVQKLLGQFSEKELAAEKKRILHCLGLAEEIQKYCCSRK'

    # gene_name = 'BLVRA'
    # protein_seq = blvra

    # gene_name = 'PRRSV-2'
    # protein_seq = 'MPNNNGKQQKRKKGDGQPVNQLCQMLGKIIAQQNQSRGKGPGKKNKKKNPEKPHFPLATEDDVRHHFTPSERQLCLSSIQTAFNQGAGTCTLSDSGRISYTVEFSLPTHHTVRLIRVTASPSA'
    # gene_name = 'PRRSV-2_half_1'
    # protein_seq = 'MPNNNGKQQKRKKGDGQPVNQLCQMLGKIIAQQNQSRGKGPGKKNKKKNPEKPHFPLATEDDVRHHFTPSER'
    # gene_name = 'PRRSV-2_half_2'
    # protein_seq = 'QLCLSSIQTAFNQGAGTCTLSDSGRISYTVEFSLPTHHTVRLIRVTASPSA'
    # gene_name = 'ENV_HV1H2_1-511'
    # protein_seq = 'MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNSNNESEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL'[:511]
    gene_name = 'ENV_HV1H2_34-511'
    protein_seq = 'MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNSNNESEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL'[33:511]
    # gene_name = 'HEMA_I68A0'
    # protein_seq = 'MKTIIALSYIFCLALGQDLPGNDNSTATLCLGHHAVPNGTLVKTITDDQIEVTNATELVQSSSTGKICNNPHRILDGIDCTLIDALLGDPHCDVFQNETWDLFVERSKAFSNCYPYDVPDYASLRSLVASSGTLEFITEGFTWTGVTQNGGSNACKRGPGSGFFSRLNWLTKSGSTYPVLNVTMPNNDNFDKLYIWGIHHPSTNQEQTSLYVQASGRVTVSTRRSQQTIIPNIGSRPWVRGLSSRISIYWTIVKPGDVLVINSNGNLIAPRGYFKMRTGKSSIMRSDAPIDTCISECITPNGSIPNDKPFQNVNKITYGACPKYVKQNTLKLATGMRNVPEKQTRGLFGAIAGFIENGWEGMIDGWYGFRHQNSEGTGQAADLKSTQAAIDQINGKLNRVIEKTNEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTRRQLRENAEEMGNGCFKIYHKCDNACIESIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISCFLLCVVLLGFIMWACQRGNIRCNICI'
    # gene_name = 'VIF_HV1N5_1-151'
    # protein_seq = 'MENRWQVMIVWQVDRMRINTWKRLVKHHMYISRKAKDWFYRHHYESTNPKISSEVHIPLGDAKLVITTYWGLHTGERDWHLGQGVSIEWRKKRYSTQVDPDLADQLIHLHYFDCFSESAIRNTILGRIVSPRCEYQAGHNKVGSLQYLALAALIKPKQIKPPLPSVRKLTEDRWNKPQKTKGHRGSHTMNGH'[:151]

    protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
    protein_seq = protein_seq.unsqueeze(0).to(device)
    
    save_file = output_dir / gene_name
    save_file.mkdir(parents=True, exist_ok=True)

    # save the input cell image
    save_colored_image((chosen_nucleus_img.squeeze() + 1) / 2, save_file / 'chosen_nucleus.png', color='blue')
    save_colored_image((chosen_microtubules_img.squeeze() + 1) / 2, save_file / 'chosen_microtubules.png', color='red')
    save_colored_image((chosen_ER_img.squeeze() + 1) / 2, save_file / 'chosen_ER.png', color='green')

    real_protein_seq = decoding.decode_sequence(protein_seq.squeeze(), vocab)
    print(real_protein_seq)
        
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start

        samples = model.sequence_to_image(
            protein_seq.repeat(current_batch_size, 1), 
            chosen_cell_img.repeat(current_batch_size, 1, 1, 1), 
            num_steps=config.num_steps, 
        )
        for j in range(current_batch_size):
            sample = samples[j]
            pred_img = sample

            save_image(pred_img, save_file / f'generated_img_{batch_start + j}.png', normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()



