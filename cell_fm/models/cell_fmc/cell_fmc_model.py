import torch
import torch.nn as nn
import numpy as np

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers import SD3Transformer2DModel

from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel as BaseCELLFMModel
from cell_fm.models.cell_fm.modules import TimestepEmbedder, CondConvNet
from cell_fm.models.cell_fm.modules.positional_embedding import get_1d_sincos_pos_embed
from cell_fm.models.cell_fm.modules.protein_sequence_embedding import ESMEmbed
from cell_fm.models.cell_fm.modules.transport import create_transport, Sampler
from cell_fm.pipeline.utils import CELLFMOutput

from .cell_fmc_config import CELLFMCConfig


class CELLFMCModel(BaseCELLFMModel):
    config_class = CELLFMCConfig

    def initialize_net(self, config: CELLFMCConfig):
        return CELLFMC(config)

    def forward(self, batched_data, **kwargs):
        protein_seq = batched_data['protein_seq']
        interaction_context = batched_data['interaction_context']  # [B, n_context, context_embedding_dim]

        protein_img_latent, cell_img = self.prepare_data(batched_data, self.vae)

        t, x0, x1 = self.transport.sample(protein_img_latent)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

        img_output = self.net(xt, cell_img, protein_seq, interaction_context, t)

        B, *_, C = xt.shape
        assert img_output.size() == (B, *xt.size()[1:-1], C)

        loss_dict = self.transport.training_losses(img_output, x0, xt, ut, t)
        img_diff_loss = loss_dict["loss"].mean()

        model_output = (img_diff_loss,)
        loss, log_loss = self.loss(batched_data, model_output)

        return CELLFMOutput(loss=loss, log_output=log_loss)

    def sequence_to_image(self, protein_seq, cell_img, interaction_context, num_steps=100):
        protein_img_latent = torch.randn(
            cell_img.shape[0],
            self.config.latent_channels,
            self.config.sample_size,
            self.config.sample_size,
            device=cell_img.device,
        )

        with torch.no_grad():
            sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

            def fn(img_latent, t):
                return self.net(img_latent, cell_img, protein_seq, interaction_context, t)

            protein_img_latent = sample_fn(protein_img_latent, fn)[-1]

        return self.vae.decode(protein_img_latent).sample


class CELLFMC(nn.Module):
    def __init__(self, config: CELLFMCConfig):
        super().__init__()
        self.config = config

        self.cond_conv = CondConvNet(
            config.in_channels * len(config.cell_image.split(',')),
            config.cond_out_channels,
        )

        self.time_embedding = TimestepEmbedder(hidden_size=config.encoder_hidden_size)

        self.img_embedding = PatchEmbed(
            height=config.sample_size,
            width=config.sample_size,
            patch_size=config.img_generator_patch_size,
            in_channels=config.latent_channels + config.cond_out_channels[-1],
            embed_dim=config.num_attention_heads * config.attention_head_dim,
            pos_embed_type=None,
        )

        # Protein sequence positional embedding
        seq_pos_embed = get_1d_sincos_pos_embed(
            config.encoder_hidden_size, config.max_protein_sequence_len + 2,
        )
        self.protein_sequence_pos_embed = nn.Parameter(
            torch.from_numpy(seq_pos_embed).float().unsqueeze(0),
            requires_grad=False,
        )

        # Interaction context positional embedding
        ctx_pos_embed = get_1d_sincos_pos_embed(
            config.encoder_hidden_size, config.n_context_proteins,
        )
        self.context_pos_embed = nn.Parameter(
            torch.from_numpy(ctx_pos_embed).float().unsqueeze(0),
            requires_grad=False,
        )

        self.seq_proj_in = nn.Linear(config.encoder_hidden_size, config.encoder_hidden_size, bias=True)
        self.context_proj = nn.Linear(config.context_embedding_dim, config.encoder_hidden_size, bias=True)

        self.img_generator = SD3Transformer2DModel(
            sample_size=config.sample_size,
            patch_size=config.img_generator_patch_size,
            in_channels=config.latent_channels + config.cond_out_channels[-1],
            num_layers=config.img_generator_num_layers,
            attention_head_dim=config.attention_head_dim,
            num_attention_heads=config.num_attention_heads,
            joint_attention_dim=config.encoder_hidden_size,
            pooled_projection_dim=config.encoder_hidden_size,
            caption_projection_dim=config.num_attention_heads * config.attention_head_dim,
            out_channels=config.latent_channels,
        )

        self.initialize_weights()
        self.initialize_protein_sequence_embedding()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def initialize_protein_sequence_embedding(self):
        self.protein_sequence_embedding = ESMEmbed(
            self.config.esm_embedding,
            self.config.encoder_hidden_size,
            self.config.esm_fixed_embedding,
        )

    def forward(self, protein_img, cell_img, protein_seq, interaction_context, time):
        time_embeds = self.time_embedding(time)  # [B, D]

        # Protein sequence embeddings
        seq_embeds = self.protein_sequence_embedding(protein_seq)  # [B, T, D]
        seq_embeds = self.seq_proj_in(seq_embeds)
        seq_embeds = seq_embeds + self.protein_sequence_pos_embed[:, :protein_seq.shape[1]]

        # Interaction context embeddings
        ctx_embeds = self.context_proj(interaction_context)  # [B, n_context, D]
        ctx_embeds = ctx_embeds + self.context_pos_embed[:, :interaction_context.shape[1]]

        # Concatenate as condition for SD3
        encoder_hidden_states = torch.cat([seq_embeds, ctx_embeds], dim=1)  # [B, T+n_context, D]

        # Cell image conditioning via conv downsampling
        cell_img_conv = self.cond_conv(cell_img)
        concat_img = torch.cat([protein_img, cell_img_conv], dim=1)  # [B, C+cond_C, H, W]

        img_output = self.img_generator(
            hidden_states=concat_img,
            timestep=time,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=time_embeds,
        ).sample

        return img_output
