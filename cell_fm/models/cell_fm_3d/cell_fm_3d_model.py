# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn

from diffusers.models.transformers import SD3Transformer2DModel
from transformers import PreTrainedModel

from cell_fm.models.cell_fm.modules.protein_sequence_embedding import ESMEmbed
from cell_fm.models.cell_fm.modules.positional_embedding import (
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
)
from cell_fm.models.cell_fm.modules.transport import create_transport, Sampler
from cell_fm.models.vae_3d.vae_3d_model import VAE3DModel
from cell_fm.models.vae_3d.vae_3d_config import VAE3DConfig
from cell_fm.pipeline.utils import CELLFMOutput
from cell_fm.logging import logger

from .cell_fm_3d_config import CELLFM3DConfig


# ---------------------------------------------------------------------------
# 3-D sincos positional embedding
# ---------------------------------------------------------------------------

def _get_3d_sincos_pos_embed(embed_dim: int, nd: int, nh: int, nw: int) -> np.ndarray:
    """Return (nd*nh*nw, embed_dim) 3-D sincos positional embedding.

    embed_dim is split roughly equally across the three axes (D, H, W),
    with any remainder assigned to W, so that all per-axis dims are even.
    """
    base = (embed_dim // 6) * 2      # largest even number ≤ embed_dim/3
    dim_d = base
    dim_h = base
    dim_w = embed_dim - 2 * base     # guaranteed even: embed_dim even, base even

    gd, gh, gw = np.meshgrid(
        np.arange(nd, dtype=np.float32),
        np.arange(nh, dtype=np.float32),
        np.arange(nw, dtype=np.float32),
        indexing='ij',
    )
    gd, gh, gw = gd.flatten(), gh.flatten(), gw.flatten()

    return np.concatenate([
        get_1d_sincos_pos_embed_from_grid(dim_d, gd),
        get_1d_sincos_pos_embed_from_grid(dim_h, gh),
        get_1d_sincos_pos_embed_from_grid(dim_w, gw),
    ], axis=1)  # (nd*nh*nw, embed_dim)


# ---------------------------------------------------------------------------
# 3-D patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """Conv3d-based patch embedding with 3-D sincos positional embedding."""

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        patch_d: int,
        patch_hw: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(patch_d, patch_hw, patch_hw),
            stride=(patch_d, patch_hw, patch_hw),
        )
        nd, nh, nw = depth // patch_d, height // patch_hw, width // patch_hw
        pos_embed = _get_3d_sincos_pos_embed(embed_dim, nd, nh, nw)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float().unsqueeze(0),  # (1, N, D)
        )
        self.num_patches = nd * nh * nw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x)                  # (B, embed_dim, nd, nh, nw)
        x = x.flatten(2).transpose(1, 2)  # (B, nd*nh*nw, embed_dim)
        return x + self.pos_embed


# ---------------------------------------------------------------------------
# SD3-based 3-D image generator
# ---------------------------------------------------------------------------

class SD3Transformer3DModel(nn.Module):
    """SD3 transformer adapted for 3-D volumetric latents.

    Builds an SD3Transformer2DModel as a factory to obtain its transformer
    blocks and conditioning modules, then replaces the 2-D patch embedding
    with a 3-D variant and overrides unpatchification in forward().
    """

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        patch_d: int,
        patch_hw: int,
        latent_channels: int,
        num_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        joint_attention_dim: int,
        pooled_projection_dim: int,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        # Build SD3 with a minimal dummy 2-D config to instantiate transformer blocks.
        # sample_size=4, patch_size=4, pos_embed_max_size=4 → tiny pos_embed buffer.
        _sd3 = SD3Transformer2DModel(
            sample_size=4,
            patch_size=4,
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            caption_projection_dim=inner_dim,
            pos_embed_max_size=4,
        )

        # Extract all reusable components and register them as proper submodules.
        self.time_text_embed = _sd3.time_text_embed
        self.context_embedder = _sd3.context_embedder
        self.transformer_blocks = _sd3.transformer_blocks
        self.norm_out = _sd3.norm_out
        del _sd3  # free the dummy pos_embed + proj_out

        # 3-D patch embedding (replaces SD3's pos_embed)
        self.pos_embed_3d = PatchEmbed3D(
            depth, height, width, patch_d, patch_hw, latent_channels, inner_dim
        )

        # Output projection: tokens → 3-D patches
        self.proj_out = nn.Linear(inner_dim, patch_d * patch_hw * patch_hw * latent_channels)

        self._patch_d = patch_d
        self._patch_hw = patch_hw
        nd, nh, nw = depth // patch_d, height // patch_hw, width // patch_hw
        self._nd, self._nh, self._nw = nd, nh, nw
        self._out_channels = latent_channels

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (B, C, D, H, W)
        hidden_states = self.pos_embed_3d(hidden_states)                         # (B, N, D_model)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)                             # (B, N, pd*ph*pw*C)

        # 3-D unpatchify
        pd, ph, pw = self._patch_d, self._patch_hw, self._patch_hw
        nd, nh, nw = self._nd, self._nh, self._nw
        B, C = hidden_states.shape[0], self._out_channels

        hidden_states = hidden_states.reshape(B, nd, nh, nw, pd, ph, pw, C)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)          # (B, C, nd, pd, nh, ph, nw, pw)
        output = hidden_states.reshape(B, C, nd * pd, nh * ph, nw * pw)         # (B, C, D, H, W)
        return output


# ---------------------------------------------------------------------------
# CELLFM3D — inner network (no VAE, no transport)
# ---------------------------------------------------------------------------

class CELLFM3D(nn.Module):

    def __init__(self, config: CELLFM3DConfig):
        super().__init__()

        D, H, W = config.input_spatial_size
        # Latent spatial size: (num_down_blocks - 1) actual downsamples → factor 2^(n-1)
        n_ds = config.num_down_blocks - 1
        d_lat = D // (2 ** n_ds)
        h_lat = H // (2 ** n_ds)
        w_lat = W // (2 ** n_ds)

        # ESM sequence embedding
        self.protein_sequence_embedding = ESMEmbed(
            config.esm_embedding,
            config.encoder_hidden_size,
            config.esm_fixed_embedding,
        )
        self.seq_proj_in = nn.Linear(config.encoder_hidden_size, config.encoder_hidden_size)

        protein_sequence_pos_embed = get_1d_sincos_pos_embed(
            config.encoder_hidden_size, config.max_protein_sequence_len + 2
        )
        self.protein_sequence_pos_embed = nn.Parameter(
            torch.from_numpy(protein_sequence_pos_embed).float().unsqueeze(0),
            requires_grad=False,
        )

        # 3-D SD3 image generator
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

    def forward(
        self,
        protein_img_latent: torch.Tensor,
        protein_seq: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        # Sequence embedding
        seq_embeds = self.protein_sequence_embedding(protein_seq)               # (B, T, hidden)
        seq_embeds = self.seq_proj_in(seq_embeds)
        seq_embeds = seq_embeds + self.protein_sequence_pos_embed[:, :protein_seq.shape[1]]

        # Mean-pooled sequence as global conditioning (replaces SD3's pooled text emb)
        pooled = seq_embeds.mean(dim=1)                                         # (B, hidden)

        # Flow matching velocity prediction
        img_output = self.img_generator(
            hidden_states=protein_img_latent,
            encoder_hidden_states=seq_embeds,
            pooled_projections=pooled,
            timestep=time,
        )
        return img_output


# ---------------------------------------------------------------------------
# CELLFM3DModel — top-level PreTrainedModel with frozen VAE3D
# ---------------------------------------------------------------------------

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

        self.vae = self._build_vae(config)
        self.load_pretrained_weights(config, checkpoint_path=config.loadcheck_path)
        self.vae = self._load_and_freeze_vae(self.vae, config.vae_loadcheck_path)

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    def _build_vae(self, config: CELLFM3DConfig) -> VAE3DModel:
        vae_config = VAE3DConfig(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            num_down_blocks=config.num_down_blocks,
            latent_channels=config.latent_channels,
            vae_block_out_channels=','.join(map(str, config.vae_block_out_channels)),
            input_spatial_size=','.join(map(str, config.input_spatial_size)),
            norm_num_groups=config.norm_num_groups,
            layers_per_block=config.layers_per_block,
            ft=False,
            infer=False,
            vae_loadcheck_path="",
        )
        return VAE3DModel(vae_config)

    def _load_and_freeze_vae(self, vae: VAE3DModel, checkpoint_path: str) -> VAE3DModel:
        if checkpoint_path and checkpoint_path != '.':
            vae_config = deepcopy(vae.config)
            vae_config.ft = False
            vae_config.infer = True
            vae.load_pretrained_weights(vae_config, checkpoint_path=checkpoint_path)
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        return vae

    # ------------------------------------------------------------------
    # Checkpoint loading (CELL-FM pattern)
    # ------------------------------------------------------------------

    def load_pretrained_weights(self, config: CELLFM3DConfig, checkpoint_path: str):
        if not (config.ft or config.infer):
            return

        logger.info(f"{'Finetune' if config.ft else 'Infer'} from checkpoint: {checkpoint_path}")

        if os.path.isdir(checkpoint_path):
            index_path = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
            with open(index_path) as f:
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

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, batched_data, **kwargs):
        protein_img = batched_data['protein_img']           # (B, 1, D, H, W)
        protein_seq = batched_data['protein_seq_masked']    # (B, T)

        with torch.no_grad():
            protein_img_latent = self.vae.encode(protein_img).sample()  # (B, 4, d, h, w)

        t, x0, x1 = self.transport.sample(protein_img_latent)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

        img_output = self.net(xt, protein_seq, t)

        loss = self.transport.training_losses(img_output, x0, xt, ut, t)["loss"].mean()
        return CELLFMOutput(loss=loss)

    # ------------------------------------------------------------------
    # Inference: sequence → 3-D protein image
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sequence_to_image(self, protein_seq: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        device = protein_seq.device
        B = protein_seq.shape[0]

        # Latent spatial size
        n_ds = self.config.num_down_blocks - 1
        D, H, W = self.config.input_spatial_size
        d_lat = D // (2 ** n_ds)
        h_lat = H // (2 ** n_ds)
        w_lat = W // (2 ** n_ds)

        noise = torch.randn(B, self.config.latent_channels, d_lat, h_lat, w_lat, device=device)

        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        def model_fn(xt, t):
            return self.net(xt, protein_seq, t)

        latent = sample_fn(noise, model_fn)[-1]
        return self.vae.decode(latent).sample
