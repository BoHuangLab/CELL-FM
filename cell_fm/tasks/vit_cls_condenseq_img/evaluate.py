# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.data.condenseq_img_data.dataset import CondenSeqImageDataset
from cell_fm.data.condenseq_img_data.config import CondenSeqImageDatasetConfig
from cell_fm.models.vit_cls_condenseq_img.config import ViTConfig
from cell_fm.models.vit_cls_condenseq_img.model import ViTModel
from cell_fm.utils.cli_utils import cli

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


@cli(CondenSeqImageDatasetConfig, ViTConfig)
def main(args) -> None:
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config & data
    config = ViTConfig(**vars(args))
    dataconfig = CondenSeqImageDatasetConfig(**vars(args))
    testset = CondenSeqImageDataset(dataconfig, split_key=dataconfig.split_key)

    val_loader = DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=testset.collate,
        drop_last=False,
    )

    # model
    model = ViTModel(config=config)
    model.to(device)
    model.eval()

    # collect all preds / labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Processing valset"):
            data = data["batched_data"]

            label = data["label"]                        # [B]
            protein_img = data["protein_img"].to(device) # [B, 1, H, W]
            nucleus_img = data["nucleus_img"].to(device) # [B, 1, H, W]
            input_img = torch.cat([nucleus_img, protein_img], dim=1)  # [B, 2, H, W] 

            logits = model.predict(input_img)  # [B, num_classes]
            preds = torch.argmax(logits, dim=1).cpu()         # [B]

            all_preds.append(preds)
            all_labels.append(label.cpu())

    # stack to 1D arrays
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # figure out which classes exist (should be 2)
    classes_sorted = np.unique(all_labels)
    # e.g. array([0, 1]) or array([2, 3])

    # overall accuracy
    acc = accuracy_score(all_labels, all_preds)

    # precision/recall/f1
    # average=None -> per-class in the order of classes_sorted
    prec_per_class, rec_per_class, f1_per_class, support_per_class = \
    precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=classes_sorted,
        average=None,
        zero_division=0
    )

    # binary / macro-style summaries
    # - "binary" only makes sense if sklearn can infer positive_label,
    #   so to be safe we report macro (just mean of the 2 classes)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=classes_sorted,
        average="macro",
        zero_division=0
    )

    # confusion matrix (2x2 expected)
    cm = confusion_matrix(all_labels, all_preds, labels=classes_sorted)

    # ----- print nicely -----
    print("\n=== Overall Metrics ===")
    print(f"Accuracy: {acc:.6f} ({acc*100:.2f}%)")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall:    {rec_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")

    print("\n=== Per-class Metrics ===")
    for idx_in_order, cls_id in enumerate(classes_sorted):
        p = prec_per_class[idx_in_order]
        r = rec_per_class[idx_in_order]
        f1 = f1_per_class[idx_in_order]
        sup = support_per_class[idx_in_order]
        print(f"Class {cls_id}: "
              f"Prec={p:.4f}  Rec={r:.4f}  F1={f1:.4f}  Support={sup}")

    print("\n=== Confusion Matrix ===")
    print("rows = true class, cols = predicted class")
    print("class order:", classes_sorted)
    print(cm)

    # optional detailed classification_report (still handy for logs)
    print("\n=== classification_report ===")
    target_names = [f"class {c}" for c in classes_sorted]
    print(classification_report(
        all_labels,
        all_preds,
        labels=classes_sorted,
        target_names=target_names,
        zero_division=0
    ))


if __name__ == "__main__":
    main()
