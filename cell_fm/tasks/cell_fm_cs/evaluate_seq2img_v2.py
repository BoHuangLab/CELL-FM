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

from torchvision.utils import save_image
import tifffile as tiff
import numpy as np

from esm.utils import encoding



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

def save_tif(image, output_path):
    # image = np.transpose(image, (1, 0, 2, 3))
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

    # DDX4
    # protein_name = 'DDX4'
    # protein_seq = 'EDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFG'

    # protein_name = 'DDX4_blocky'
    # protein_seq = 'RRNPTKNRGFSRRGGYRRGNNSRASGPYRRGGEGSFDGCDGGFGLGSPNNELDPDDCMQETGGLFG'

    # protein_name = 'DDX4_m1'
    # protein_seq = 'RRNPTKNRGRSRRGGRRRGNNSRASGPRRRGGEGSDDGCDGGDGLGSPNNELDPDDCMQETGGLDG'

    # protein_name = 'DDX4_m2'
    # protein_seq = 'RRNPTKNRGFSRRGGFRRGNNSRASGPFRRGGEGSFDGCDGGFGLGSPNNELDPDDCMQETGGLFG'

    # protein_name = 'DDX4_m3'
    # protein_seq = 'RRNPTKNRGGSRRGGGRRGNNSRASGPGRRGGEGSGDGCDGGGGLGSPNNELDPDDCMQETGGLGG'

    # protein_name = 'DDX4_m4'
    # protein_seq = 'RRNGGKNRGGSRRGGGRRGNNSRGSGGGRRGGEGSGDGCDGGGGGGSGNNEGDGDDCMGEGGGGGG'

    # protein_name = 'DDX4_m5' # DDX4_blocky, G to P.
    # protein_seq = 'RRNPTKNRPFSRRPPYRRPNNSRASPPYRRPPEPSFDPCDPPFPLPSPNNELDPDDCMQETPPLFP'

    # protein_name = 'DDX4_m6' # DDX4_blocky, G to T, S to T.
    # protein_seq = 'RRNPTKNRTFTRRTTYRRTNNTRATTPYRRTTETTFDTCDTTFTLTTPNNELDPDDCMQETTTLFT'

    # protein_name = 'DDX4_m7' # DDX4_blocky, G to P, S to P, A to P.
    # protein_seq = 'RRNPTKNRPFPRRPPYRRPNNPRPPPPYRRPPEPPFDPCDPPFPLPPPNNELDPDDCMQETPPLFP'

    # protein_name = 'DDX4_m8' # DDX4_m7, E to D, K to R.
    # protein_seq = 'RRNPTRNRPFPRRPPYRRPNNPRPPPPYRRPPDPPFDPCDPPFPLPPPNNDLDPDDCMQDTPPLFP'

    # protein_name = 'DDX4_m9' # DDX4_m8, change some P to Y.
    # protein_seq = 'RRNPTRNRPFPRRPPYRRPNNPRYPPYYRRPPDPPFDPCDPPFPLYPPNNDLDPDDCMQDTYPLFP'

    # protein_name = 'DDX4_m10' # DDX4_m9, change 1R to D, 1R to Y.
    # protein_seq = 'RRNPTRNRPFPRRPPYRRPNNPRYPPYYYDPPDPPFDPCDPPFPLYPPNNDLDPDDCMQDTYPLFP'

    # protein_name = 'DDX4_m11' # DDX4_m10. change Q, T, N to P.
    # protein_seq = 'RRPPPRPRPFPRRPPYRRPPPPRYPPYYYDPPDPPFDPCDPPFPLYPPPPDLDPDDCMPDPYPLFP'

    # protein_name = 'DDX4_m12' # DDX4_m11, change D,R to Y. 
    # protein_seq = 'YYNPTYNYPFPYYPPYYYPNNPYYPPYYYYPPYPPFYPCYPPFPLYPPNNYLYPYYCMQYTYPLFP'

    # protein_name = 'DDX4_m13' # DDX4_m12. F to Y.
    # protein_seq = 'YYNPTYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPCYPPYPLYPPNNYLYPYYCMQYTYPLYP'

    # protein_name = 'DDX4_m14' # DDX4_m13. Q, T, N to P.
    # protein_seq = 'YYPPPYPYPYPYYPPYYYPPPPYYPPYYYYPPYPPYYPCYPPYPLYPPPPYLYPYYCMPYPYPLYP'

    # protein_name = 'DDX4_m15' # DDX4_m13. P to T.
    # protein_seq = 'YYNTTYNYTYTYYTTYYYTNNTYYTTYYYYTTYTTYYTCYTTYTLYTTNNYLYTYYCMQYTYTLYT'

    # protein_name = 'DDX4_m16' # DDX4_m13. 2Y to 2N.
    # protein_seq = 'YYNPTYNYPYPYYPPYYYPNNPYYPPYYNNPPYPPYYPCYPPYPLYPPNNYLYPYYCMQYTYPLYP'

    # protein_name = 'DDX4_m17' # DDX4_m13. C,M to Y
    # protein_seq = 'YYNPTYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPYYPPYPLYPPNNYLYPYYYYQYTYPLYP'

    # protein_name = 'DDX4_m18' # DDX4_m17. L to Y
    # protein_seq = 'YYNPTYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPYYPPYPYYPPNNYYYPYYYYQYTYPYYP'

    # protein_name = 'DDX4_m19' # DDX4_m18. N to P
    # protein_seq = 'YYPPTYPYPYPYYPPYYYPPPPYYPPYYYYPPYPPYYPYYPPYPYYPPPPYYYPYYYYQYTYPYYP'

    # protein_name = 'DDX4_m20' # DDX4_m18. T to P # Best so far Max Gap: 0.5430
    # protein_seq = 'YYNPPYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPYYPPYPYYPPNNYYYPYYYYQYPYPYYP'

    # protein_name = 'DDX4_m21' # DDX4_m20. Q to P
    # protein_seq = 'YYNPPYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPYYPPYPYYPPNNYYYPYYYYPYPYPYYP'

    # protein_name = 'DDX4_m22' # DDX4_m20. Q to Y
    # protein_seq = 'YYNPPYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPYYPPYPYYPPNNYYYPYYYYYYPYPYYP'

    # protein_name = 'DDX4_m23' # DDX4_m20. 1Y to 1Q
    # protein_seq = 'YYNPPYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPYYPPYPYYPPNNYQYPYYYYQYPYPYYP'

    # protein_name = 'DDX4_m24' # DDX4_m20. Q to N
    # protein_seq = 'YYNPPYNYPYPYYPPYYYPNNPYYPPYYYYPPYPPYYPYYPPYPYYPPNNYYYPYYYYNYPYPYYP'

    # protein_name = 'DDX4_m25' # DDX4_m20. P to G
    # protein_seq = 'YYNGGYNYGYGYYGGYYYGNNGYYGGYYYYGGYGGYYGYYGGYGYYGGNNYYYGYYYYQYGYGYYG'

    # protein_name = 'DDX4_m26' # DDX4_m25. some Y to G
    # protein_seq = 'YYNGGYNYGYGYYGGYGGGNNGYYGGYYGGGGYGGYYGYYGGYGYYGGNNYYYGGGYYQYGYGYYG'

    # protein_name = 'YG_m1' # 11 YGGGGG
    # protein_seq = 'YGGGGG' * 11

    # protein_name = 'YGN_m1' # 11 YGGGGN
    # protein_seq = 'YGGGGN' * 11

    # protein_name = 'YGN_m2' #
    # # protein_seq = 'NGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGG'
    # protein_seq = 'NGYGNNGGYGGGGGGYNGGNYGYGNGGNGYGNNGNGYGGNGNYNNYNNGGGGGYGGGNGNNYGGGG'

    # protein_name = 'YGN_m3' # 11 YGGGNN
    # protein_seq = 'YGGGNN' * 11

    # protein_name = 'YGN_m4'
    # protein_seq = 'YGGGGGGYGGGGGGYGGGGGGYGGGGGGYGGGGGGYGGGNNYNNNNYNNNNYNNNNYNNNNYNNNN'

    # print('number of Y:', protein_seq.count('Y'))
    # print('number of G:', protein_seq.count('G'))
    # print('number of N:', protein_seq.count('N'))

    # S/Q/N/E/D/K/R
    # 'NGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGG'
    # 'YGGGGNYGGGGNYGGGGNYGGGGNYGGGGNYGGGGNYGGGGNYGGGGNYGGGGNYGGGGNYGGGGN'

    # protein_name = 'RGRGG'
    # protein_seq = 'RGRGG' * 6 + "N" * 6 + "DGDNG" * 6

    # protein_name = 'RGRGG_R1K'
    # protein_seq = 'KGKGG' + 'RGRGG' * 5 + "N" * 6 + "DGDNG" * 6

    protein_name = 'RGRGG_R1K_R2K'
    protein_seq = 'KGKGG' + 'KGKGG' + 'RGRGG' * 4 + "N" * 6 + "DGDNG" * 6

    # protein_name = 'KGKGG'
    # protein_seq = 'KGKGG' * 6 + "N" * 6 + "DGDNG" * 6

    output_dir = Path(config.output_dir)
    output_dir = output_dir / protein_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # chosen images
    protein_index = 12626
    index = 0

    valset = CondenSeqAllImageDataset(args, split_key='all')
    valset.meta_data = valset.meta_data[valset.meta_data['index'] == protein_index]
    chosen_data = valset.__getitem__(0)

    vocab = valset.vocab

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_cell_img = chosen_nucleus_img

    max_protein_intensity_level = 4096
    num_protein_intensity_levels = 4096
    batch_size = 512

    protein_intensity_levels_linspace = torch.linspace(0, max_protein_intensity_level, steps=num_protein_intensity_levels).to(device)

    print(protein_seq)

    protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
    protein_seq = protein_seq.unsqueeze(0).to(device)

    generated_imgs = []

    for batch_start in range(0, num_protein_intensity_levels, batch_size):
        batch_end = min(batch_start + batch_size, num_protein_intensity_levels)
        current_batch_size = batch_end - batch_start

        protein_intensity_level_batch = protein_intensity_levels_linspace[batch_start:batch_end]
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

    # save protein_intensity_levels_linspace as npy
    np.save(
        output_dir / "protein_intensity_levels.npy",
        protein_intensity_levels_linspace.cpu().numpy(),
    )

    logger.info("Done!")

if __name__ == "__main__":
    main()