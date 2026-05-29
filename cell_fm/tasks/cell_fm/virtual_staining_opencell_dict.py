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

from torchvision.utils import save_image
import numpy as np
import tifffile as tiff
from esm.utils import encoding, decoding

def save_tif(image, output_path):
    tensor_np = image.cpu().numpy()
    tensor_np = tensor_np.clip(0, 1)
    tensor_np = np.round(tensor_np * 65535)

    tensor_np = tensor_np.astype(np.uint16)
    tiff.imwrite(output_path, tensor_np, imagej=True)

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
    valset = OpenCellCropAllImageDataset(config, split_key=config.split_key)
    vocab = valset.vocab
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # chosen images
    gene_name = 'ATG7'
    index = 1

    valset.meta_data = valset.meta_data[valset.meta_data['gene_name'] == gene_name]
    chosen_data = valset.__getitem__(0)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)

    chosen_cell_img = chosen_nucleus_img

    # number of samples
    num_samples = 32

    # batch size
    batch_size = 32

    # wt_name = "PRRSV-win-72"
    # wt_seq  = 'MPNNNGKQQKRKKGDGQPVNQLCQMLGKIIAQQNQSRGKGPGKKNKKKNPEKPHFPLATEDDVRHHFTPSERQLCLSSIQTAFNQGAGTCTLSDSGRISYTVEFSLPTHHTVRLIRVTASPSA'
    # window  = 72

    # seq_dict = {}
    # for i in range(len(wt_seq) - window + 1):
    #     mut_seq  = wt_seq[:i] + wt_seq[i + window:]
    #     mut_name = f"del{i+1}-{i+window}"
    #     seq_dict[mut_name] = mut_seq

    # wt_name = "H2BC21"
    # wt_seq  = 'MPEPAKSAPAPKKGSKKAVTKAQKKDGKKRKRSRKESYSIYVYKVLKQVHPDTGISSKAMGIMNSFVNDIFERIAGEASRLAHYNKRSTITSREIQTAVRLLLPGELAKHAVSEGTKAVTKYTSSK'

    # window  = 20

    # seq_dict = {}
    # for i in range(1, len(wt_seq) - window + 1):
    #     mut_seq  = wt_seq[:i] + wt_seq[i + window:]
    #     mut_name = f"del{i+1}-{i+window}"
    #     seq_dict[mut_name] = mut_seq

    # wt_name = 'DNAJC8'
    # seq_dict = {
    #     # 'DNAJC8': 'MAASGESGTSGGGGSTEEAFMTFYSEVKQIEKRDSVLTSKNQIERLTRPGSSYFNLNPFEVLQIDPEVTDEEIKKRFRQLSILVHPDKNQDDADRAQKAFEAVDKAYKLLLDQEQKKRALDVIQAGKEYVEHTVKERKKQLKKEGKPTIVEEDDPELFKQAVYKQTMKLFAELEIKRKEREAKEMHERKRQREEEIEAQEKAKREREWQKNFEESRDGRVDSWRNFQANTKGKKEKKNRTFLRPPKVKMEQRE',
    #     # 'DNAJC8-cut': 'MAASGESGTSGGGGSTEEAFMTFYSEVKQIEKRDSVLTSKNQIERLTRPGSSYFNLNPFEVLQIDPEVTDEEIKKRFRQLSILVHPDKNQDDADRAQKAFEAVDKAYKLLLDQEQKKRALDVIQAGKEYVEHTVKERKKQLKKEGKPTIVEEDDPELFKQAVYKQTMKLFAELEIEWQKNFEESRDGRVDSWRNFQANTKGKKEKKNRTFLRPPKVKMEQRE',
    #     # 'DNAJC8-cut2': 'MSVLTSKNQIERLTRPGSSYFNLNPFEVLQIDPEVTDEEIKKRFRQLSILVHPDKNQDDADRAQKAFEAVDKAYKLLLDQEQKKRALDVIQAGKEYVEHTVKERKKQLKKEGKPTIVEEDDPELFKQAVYKQTMKLFAELEIKRKEREAKEMHERKRQREEEIEAQEKAKREREWQKNFEESRDGRVDSWRNFQANTKGKKEKKNRTFLRPPKVKMEQRE',
    #     'DNAJC8-cut3': 'MLVHPDKNQDDADRAQKAFEAVDKAYKLLLDQEQKKRALDVIQAGKEYVEHTVKERKKQLKKEGKPTIVEEDDPELFKQAVYKQTMKLFAELEIKRKEREAKEMHERKRQREEEIEAQEKAKREREWQKNFEESRDGRVDSWRNFQANTKGKKEKKNRTFLRPPKVKMEQRE',
    # }

    wt_name = 'PRRSV-section-10'
    wt_seq  = 'MPNNNGKQQKRKKGDGQPVNQLCQMLGKIIAQQNQSRGKGPGKKNKKKNPEKPHFPLATEDDVRHHFTPSERQLCLSSIQTAFNQGAGTCTLSDSGRISYTVEFSLPTHHTVRLIRVTASPSA'
    window  = 10

    seq_dict = {}
    for i in range(1, len(wt_seq) - window + 1):
        mut_seq  = "M" + wt_seq[i:i+window]
        mut_name = f"{i+1}-{i+window}"
        seq_dict[mut_name] = mut_seq

    for prot_name, prot_seq in seq_dict.items():
        print(prot_seq)
        protein_seq = encoding.tokenize_sequence(prot_seq, vocab, True).unsqueeze(0).to(device)

        save_file = output_dir / wt_name / prot_name
        save_file.mkdir(parents=True, exist_ok=True)

        print(f"[{prot_name}]  len={protein_seq.size(1)}")

        for batch_start in range(0, num_samples, batch_size):
            batch_end         = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start

            samples = model.sequence_to_image(
                protein_seq.repeat(current_batch_size, 1),
                chosen_cell_img.repeat(current_batch_size, 1, 1, 1),
                num_steps=config.num_steps,
            )
            for j in range(current_batch_size):
                save_image(samples[j], save_file / f'generated_img_{batch_start + j}.png',
                           normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()