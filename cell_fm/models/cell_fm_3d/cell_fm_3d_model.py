# -*- coding: utf-8 -*-
import os
import json
from copy import deepcopy

import torch
import torch.nn as nn

from transformers import PreTrainedModel

from cell_fm.models.cell_fm.modules.protein_sequence_embedding import ESMEmbed
from cell_fm.models.cell_fm.modules.positional_embedding import get_1d_sincos_pos_embed
from cell_fm.models.cell_fm.modules.transport import create_transport, Sampler
from cell_fm.models.vae_3d.vae_3d_model import VAE3DModel
from cell_fm.models.vae_3d.vae_3d_config import VAE3DConfig
from cell_fm.pipeline.utils import CELLFMOutput
from cell_fm.logging import logger

from .cell_fm_3d_config import CELLFM3DConfig
from .modules import SD3Transformer3DModel


class CELLFM3DModel(PreTrainedModel):
    config_class = CELLFM3DConfig

    def __init__(self, config: CELLFM3DConfig):
        super().__init__(config)
        self.config = config

        self.net = CELLFM3D(config)

        self.transport = create_transport(
            config.path_type,
            config.prediction,
            config.loss_weight,
            config.train_eps,
            config.sample_eps,
        )
        self.transport_sampler = Sampler(self.transport)

        self.vae = self.initialize_vae(config)
        self.load_pretrained_weights(config, checkpoint_path=config.loadcheck_path)
        self.vae = self.prepare_vae(self.vae, config.vae_loadcheck_path)

    def initialize_vae(self, config: CELLFM3DConfig) -> VAE3DModel:
        vae_config = VAE3DConfig(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            num_down_blocks=config.num_down_blocks,
            latent_channels=config.latent_channels,
            vae_block_out_channels=config.vae_block_out_channels,
            input_spatial_size=config.input_spatial_size,
            norm_num_groups=config.norm_num_groups,
            layers_per_block=config.layers_per_block,
            ft=False,
            infer=False,
            vae_loadcheck_path="",
        )
        return VAE3DModel(vae_config)

    def prepare_vae(self, vae: VAE3DModel, checkpoint_path: str) -> VAE3DModel:
        if checkpoint_path and checkpoint_path != '.':
            vae_config = deepcopy(vae.config)
            vae_config.ft = False
            vae_config.infer = True
            vae.load_pretrained_weights(vae_config, checkpoint_path=checkpoint_path)
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        return vae

    def load_pretrained_weights(self, config: CELLFM3DConfig, checkpoint_path: str):
        if not (config.ft or config.infer):
            return

        if config.ft:
            logger.info(f"Finetune from checkpoint: {checkpoint_path}")
        else:
            logger.info(f"Infer from checkpoint: {checkpoint_path}")

        if os.path.isdir(checkpoint_path):
            with open(os.path.join(checkpoint_path, "pytorch_model.bin.index.json")) as f:
                weight_map = json.load(f)["weight_map"]
            shard_weights = {}
            for shard_file in set(weight_map.values()):
                shard_weights[shard_file] = torch.load(
                    os.path.join(checkpoint_path, shard_file), map_location="cpu"
                )
            checkpoints_state = {k: shard_weights[v][k] for k, v in weight_map.items()}
        elif checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoints_state = load_file(checkpoint_path)
        else:
            checkpoints_state = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        model_state_dict = self.state_dict()
        filtered = {
            k: v for k, v in checkpoints_state.items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }
        keys = self.load_state_dict(filtered, strict=False)._asdict()

        missing = [k for k in keys["missing_keys"] if "dummy" not in k]
        unexpected = [k for k in keys["unexpected_keys"] if "dummy" not in k]
        if missing:
            logger.info(f"Missing keys in {checkpoint_path}: {missing}")
        if unexpected:
            logger.info(f"Unexpected keys in {checkpoint_path}: {unexpected}")

    @torch.no_grad()
    def prepare_data(self, batched_data):
        protein_img = batched_data['protein_img']               # (B, 1, D, H, W)
        protein_img_latent = self.vae.encode(protein_img).sample()
        return protein_img_latent

    def forward(self, batched_data, **kwargs):
        protein_seq = batched_data['protein_seq']
        protein_img_latent = self.prepare_data(batched_data)

        t, x0, x1 = self.transport.sample(protein_img_latent)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

        img_output = self.net(xt, protein_seq, t)
        loss = self.transport.training_losses(img_output, x0, xt, ut, t)["loss"].mean()

        log_output = {
            "train_loss": loss.item(),
        }
        return CELLFMOutput(loss=loss, log_output=log_output)

    @torch.no_grad()
    def sequence_to_image(self, protein_seq: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        device = protein_seq.device
        B = protein_seq.shape[0]

        # Derive latent spatial size: (num_down_blocks - 1) actual downsamples
        n_ds = self.config.num_down_blocks - 1
        D, H, W = self.config.input_spatial_size
        d_lat = D // (2 ** n_ds)
        h_lat = H // (2 ** n_ds)
        w_lat = W // (2 ** n_ds)

        noise = torch.randn(
            B, self.config.latent_channels, d_lat, h_lat, w_lat, device=device
        )
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        def model_fn(xt, t):
            return self.net(xt, protein_seq, t)

        latent = sample_fn(noise, model_fn)[-1]
        return self.vae.decode(latent).sample


class CELLFM3D(nn.Module):

    def __init__(self, config: CELLFM3DConfig):
        super().__init__()

        D, H, W = config.input_spatial_size
        n_ds = config.num_down_blocks - 1
        d_lat = D // (2 ** n_ds)
        h_lat = H // (2 ** n_ds)
        w_lat = W // (2 ** n_ds)

        # Image generator (3-D SD3 transformer)
        self.img_generator = SD3Transformer3DModel(
            depth=d_lat,
            height=h_lat,
            width=w_lat,
            patch_d=config.patch_d,
            patch_hw=config.patch_size,
            latent_channels=config.latent_channels,
            num_layers=config.img_generator_num_layers,
            attention_head_dim=config.attention_head_dim,
            num_attention_heads=config.num_attention_heads,
            joint_attention_dim=config.encoder_hidden_size,
            pooled_projection_dim=config.encoder_hidden_size,
        )

        # Sequence embedding
        self.initialize_protein_sequence_embedding(config)
        self.seq_proj_in = nn.Linear(config.encoder_hidden_size, config.encoder_hidden_size)

        protein_sequence_pos_embed = get_1d_sincos_pos_embed(
            config.encoder_hidden_size, config.max_protein_sequence_len + 2
        )
        self.protein_sequence_pos_embed = nn.Parameter(
            torch.from_numpy(protein_sequence_pos_embed).float().unsqueeze(0),
            requires_grad=False,
        )

    def initialize_protein_sequence_embedding(self, config: CELLFM3DConfig):
        self.protein_sequence_embedding = ESMEmbed(
            config.esm_embedding,
            config.encoder_hidden_size,
            config.esm_fixed_embedding,
        )

    def forward(
        self,
        protein_img_latent: torch.Tensor,
        protein_seq: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        seq_embeds = self.protein_sequence_embedding(protein_seq)               # (B, T, hidden)
        seq_embeds = self.seq_proj_in(seq_embeds)
        seq_embeds = seq_embeds + self.protein_sequence_pos_embed[:, :protein_seq.shape[1]]

        pooled = seq_embeds.mean(dim=1)                                         # (B, hidden)

        return self.img_generator(
            hidden_states=protein_img_latent,
            encoder_hidden_states=seq_embeds,
            pooled_projections=pooled,
            timestep=time,
        )
