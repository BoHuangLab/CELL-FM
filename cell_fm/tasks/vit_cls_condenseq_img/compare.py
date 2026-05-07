import json
from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

def _coerce_label(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        sx = x.strip().lower()
        if sx in ("true", "yes", "y", "1"):
            return 1
        if sx in ("false", "no", "n", "0"):
            return 0
        return x
    return x

def main() -> None:

    pred_name = 'PT_CondenSeq_CELL-Diff2_Rep_Dev_GFP_e_4_ignl_4_ignah_8_S1_R1_50k'
    pred_results_files = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_exp_level/{pred_name}/cellfm_test/*/decision_real.json'))
    real_results_files = sorted(glob('/hpc/reference/opencell/condenseq/vit_cls_img/eval_with_exp_level/cellfm_test/*/decision_real.json'))

    assert len(pred_results_files) == len(real_results_files), \
        f"Pred files ({len(pred_results_files)}) != Real files ({len(real_results_files)})"
    
    y_true, y_pred = [], []

    for real_file, pred_file in tqdm(zip(real_results_files, pred_results_files), total=len(real_results_files)):
        with open(real_file, 'r') as fr, open(pred_file, 'r') as fp:
            real_results = json.load(fr)
            pred_results = json.load(fp)

        real = _coerce_label(real_results.get('condenses'))
        pred = _coerce_label(pred_results.get('condenses'))

        y_true.append(real)
        y_pred.append(pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred], axis=0))

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print(f"Accuracy:           {acc:.4f}  ({int((y_true==y_pred).sum())}/{len(y_true)})")
    print(f"Balanced Accuracy:  {bacc:.4f}")
    print(f"Macro  Precision:   {prec_macro:.4f}   Recall: {rec_macro:.4f}   F1: {f1_macro:.4f}")
    print(f"Weighted Precision: {prec_weighted:.4f} Recall: {rec_weighted:.4f} F1: {f1_weighted:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"Cohen's Kappa:      {kappa:.4f}\n")

    print("Per-class metrics (aligned with classes order):")
    for i, c in enumerate(classes):
        print(f"  Class {c}:  Prec={prec_cls[i]:.4f}  Rec={rec_cls[i]:.4f}  F1={f1_cls[i]:.4f}  Support={sup_cls[i]}")

    print("\nConfusion Matrix (rows=true, cols=pred; order:", classes, ")")
    print(cm)

    print("\nclassification_report:")
    target_names = [str(c) for c in classes]
    print(classification_report(y_true, y_pred, labels=classes, target_names=target_names, zero_division=0))

    # out_dir = Path(".")
    # metrics_json = {
    #     "accuracy": acc,
    #     "balanced_accuracy": bacc,
    #     "macro": {"precision": prec_macro, "recall": rec_macro, "f1": f1_macro},
    #     "weighted": {"precision": prec_weighted, "recall": rec_weighted, "f1": f1_weighted},
    #     "mcc": mcc,
    #     "cohen_kappa": kappa,
    #     "per_class": {
    #         str(c): {
    #             "precision": float(prec_cls[i]),
    #             "recall": float(rec_cls[i]),
    #             "f1": float(f1_cls[i]),
    #             "support": int(sup_cls[i]),
    #         } for i, c in enumerate(classes)
    #     },
    #     "classes_order": [str(c) for c in classes],
    # }
    # (out_dir / "metrics.json").write_text(
    #     json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8"
    # )

    # np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

if __name__ == "__main__":
    main()
