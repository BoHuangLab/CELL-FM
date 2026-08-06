# -*- coding: utf-8 -*-
from dataclasses import dataclass
from transformers import PretrainedConfig


@dataclass
class CELLFM3DConfig(PretrainedConfig):
    model_type: str = 'cell_fm_3d'

    # Dataset parameters
    data_path: str = ""
    split_key: str = 'train'
    phase: str = 'train'
    data_aug: bool = False
    seq_zero_mask_ratio: float = 0.0
    seq_full_mask_ratio: float = 0.0
    pre_pad_seq: bool = False

    # Transport parameters
    path_type: str = 'Linear'
    prediction: str = 'velocity'
    loss_weight: str = None
    train_eps: float = None
    sample_eps: float = None

    # VAE parameters (3D VAE)
    in_channels: int = 1
    out_channels: int = 1
    num_down_blocks: int = 2
    latent_channels: int = 4
    vae_block_out_channels: str = '64,128'
    norm_num_groups: int = 32
    layers_per_block: int = 2

    # 3D spatial parameters
    input_spatial_size: str = '48,192,192'   # D,H,W of raw protein volume
    patch_d: int = 4                          # depth patch size in latent space
    patch_size: int = 8                       # H/W patch size in latent space

    # Cell image conditioning
    cell_image: str = 'nucl'
    cond_out_channels: str = '64'
    cell_image_ratio: float = 1.0

    # Sequence / ESM
    esm_embedding: str = 'esmc_300m'
    esm_fixed_embedding: bool = True
    encoder_hidden_size: int = 320
    max_protein_sequence_len: int = 2048

    # SD3-based image generator
    img_generator_num_layers: int = 18
    attention_head_dim: int = 64
    num_attention_heads: int = 16

    # Latent downsample (before SD3) / upsample (after SD3)
    down_channels: int = 64

    # UNet skip around SD3
    use_latent_skip: bool = False
    skip_channels: int = 64

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_warmup: bool = True
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.6666666666666666
    ema_update_after_step: int = 0

    # Checkpoint paths
    vae_loadcheck_path: str = '.'
    loadcheck_path: str = '.'
    ft: bool = False
    infer: bool = False
    ifresume: bool = False

    # Training parameters
    output_dir: str = ""
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    num_train_epochs: int = 10
    fp16: bool = False
    bf16: bool = False
    logging_dir: str = ""
    logging_steps: int = 10
    max_steps: int = -1
    warmup_steps: int = 1000
    save_steps: int = 1000

    dataloader_num_workers: int = 16
    seed: int = 6
    wandb: bool = False

    # Inference
    num_steps: int = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vae_block_out_channels = kwargs.get("vae_block_out_channels", self.vae_block_out_channels)
        if not isinstance(self.vae_block_out_channels, list):
            self.vae_block_out_channels = [int(c) for c in self.vae_block_out_channels.split(',')]

        self.input_spatial_size = kwargs.get("input_spatial_size", self.input_spatial_size)
        if not isinstance(self.input_spatial_size, list):
            self.input_spatial_size = [int(s) for s in self.input_spatial_size.split(',')]

        self.cond_out_channels = kwargs.get("cond_out_channels", self.cond_out_channels)
        if not isinstance(self.cond_out_channels, list):
            self.cond_out_channels = [int(c) for c in self.cond_out_channels.split(',')]
