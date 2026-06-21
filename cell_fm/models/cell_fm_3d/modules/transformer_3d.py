import torch
import torch.nn as nn

from diffusers.models.transformers import SD3Transformer2DModel

from .patch_embed_3d import PatchEmbed3D


class SD3Transformer3DModel(nn.Module):
    """SD3 transformer adapted for 3-D volumetric latents.

    Builds an SD3Transformer2DModel to obtain its transformer blocks and
    conditioning modules, then replaces the 2-D patch embedding with a 3-D
    variant and handles 3-D unpatchification in forward().
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
        in_channels: int = None,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = latent_channels
        inner_dim = num_attention_heads * attention_head_dim

        # Build SD3 with a minimal 2-D config to instantiate its transformer blocks.
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

        # Extract and register reusable components; frees the dummy pos_embed.
        self.time_text_embed = _sd3.time_text_embed
        self.context_embedder = _sd3.context_embedder
        self.transformer_blocks = _sd3.transformer_blocks
        self.norm_out = _sd3.norm_out
        del _sd3

        # 3-D patch embedding and output projection
        self.pos_embed = PatchEmbed3D(
            depth, height, width, patch_d, patch_hw, in_channels, inner_dim
        )
        self.proj_out = nn.Linear(
            inner_dim, patch_d * patch_hw * patch_hw * latent_channels
        )

        nd, nh, nw = depth // patch_d, height // patch_hw, width // patch_hw
        self._patch_d = patch_d
        self._patch_hw = patch_hw
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
        hidden_states = self.pos_embed(hidden_states)                            # (B, N, D_model)
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
        return hidden_states.reshape(B, C, nd * pd, nh * ph, nw * pw)           # (B, C, D, H, W)
