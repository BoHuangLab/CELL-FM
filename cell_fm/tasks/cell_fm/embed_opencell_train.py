# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellCropDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli
from torch.utils.data import DataLoader
from esm.utils import encoding

from tqdm import tqdm


class OpenCellCropImageClsDataset(OpenCellCropDataset):
    def collate(self, samples) -> dict:
        protein_img = torch.stack([s["protein_img"] for s in samples], 0)
        nucleus_img = torch.stack([s["nucleus_img"] for s in samples], 0)
        protein_idx = torch.stack([s["protein_idx"] for s in samples], 0)
        return {"batched_data": {"protein_img": protein_img, "nucleus_img": nucleus_img, "protein_idx": protein_idx}}


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

        # Stack multiple MLP layers
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim * 64, hidden_dim * 64))
            layers.append(nn.BatchNorm1d(hidden_dim * 64))
            layers.append(nn.GELU())
        self.mlp_stack = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim * 64, output_dim)

    def forward(self, x):
        x = x[:, 1:]  # Skip first token
        x = self.norm(x)
        x = self.input_proj(x)
        x = self.act(x)

        x = x.flatten(1)  # Flatten the sequence dimension
        x = self.mlp_stack(x)
        x = self.output_proj(x)
        return x

    def embed(self, x):
        x = x[:, 1:]  # Skip first token
        x = self.norm(x)
        x = self.input_proj(x)
        x = self.act(x)

        x = x.flatten(1)  # Flatten the sequence dimension
        x = self.mlp_stack(x)
        return x


def run_epoch(loader, model, classification_model, criterion, optimizer, scheduler, train=True, device="cuda", protein_seq=None):
    model.eval()
    classification_model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)

    for batch in tqdm(loader, leave=False):
        batch_data = batch['batched_data']

        protein_img = batch_data["protein_img"].to(device)
        cell_img = batch_data["nucleus_img"].to(device)
        labels = batch_data['protein_idx'].to(device)

        if protein_seq is None:
            protein_seq_batch = batch_data["protein_seq"].to(device)
        else:
            protein_seq_batch = protein_seq.repeat(protein_img.shape[0], 1).to(device)

        with torch.no_grad():
            img_feat_embed, seq_feat_embed = model.embed(
                protein_seq_batch, 
                protein_img, 
                cell_img, 
                img_mask_ratio=0, 
            )

        outputs = classification_model(img_feat_embed)
        loss = criterion(outputs, labels.view(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    if train:
        scheduler.step()

    return total_loss / total, correct / total


@cli(CELLFMConfig)
def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))
    train_dataset = OpenCellCropImageClsDataset(config, split_key=config.split_key)
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    vocab = train_dataset.vocab

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        num_workers=0, collate_fn=train_dataset.collate
    )

    classification_model = MLP(config.encoder_hidden_size, 1311).to(device)

    num_aa = 10
    protein_seq = "<mask>" * num_aa
    protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
    protein_seq = protein_seq.unsqueeze(0).to(device)

    # ----------------- loss / optimizer / scheduler -----------------
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.AdamW(classification_model.parameters(), lr=3e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("Training started...")
    EPOCHS = 200
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, model, classification_model, criterion, optimizer, scheduler, train=True, device=device, protein_seq=protein_seq)

        print(f"[{epoch:02d}/{EPOCHS}] "
            f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.3%}, ")

        # save model checkpoint every epoch
        model_checkpoint = output_dir / f"epoch{epoch:03d}.pth"
        print(f"Saving model to {model_checkpoint}")
        torch.save(classification_model.state_dict(), model_checkpoint)

if __name__ == "__main__":
    main()