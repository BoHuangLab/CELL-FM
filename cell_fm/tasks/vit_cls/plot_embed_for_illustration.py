# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

location_colors = {
    'Experiment': "#0072B2",  # blue
    'Predicted': '#D55E00',  # orange
}


def main() -> None:
    # output_dir = Path("./output/hpa/embed_cellfm_for_illustration")
    output_dir = Path("./output/hpa/embed_pups_for_illustration")

    saved_data = np.load(output_dir / "umap_embedding_data.npz")
    embedding_2d = saved_data['embeddings']
    labels_arr = saved_data['labels']

    # -------------------- plot (refined) --------------------
    palette = {
        "Experiment":location_colors["Experiment"],
        "Predicted": location_colors["Predicted"],
    }
    markers = {"Experiment": "v", "Predicted": "^"}
    sizes   = {"Experiment": 250, "Predicted": 250}

    unique_labels = [lbl for lbl in ["Experiment", "Predicted"] if lbl in set(labels_arr)]
    if not unique_labels:
        raise RuntimeError("No labels to plot. Check data generation above.")

    # plt.rcParams.update({
    #     "font.size": 11,
    #     "axes.titlesize": 14,
    #     "axes.labelsize": 12,
    #     "legend.fontsize": 10,
    #     "legend.title_fontsize": 10,
    #     "axes.spines.top": False,
    #     "axes.spines.right": False,
    # })

    # fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    # fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    # draw points (pred first, Experiment on top)
    for lbl in ["Predicted", "Experiment"]:
        if lbl not in unique_labels:
            continue
        m = (labels_arr == lbl)
        if not np.any(m):
            continue
        ax.scatter(
            embedding_2d[m, 0],
            embedding_2d[m, 1],
            s=sizes[lbl],
            c=palette[lbl],
            marker=markers[lbl],
            alpha=0.85 if lbl == "Predicted" else 0.95,
            linewidths=1.0,
            edgecolors="white",
            zorder=3 if lbl == "Experiment" else 2,
            rasterized=True,
        )

    # bounds
    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
    x_pad = 0.04 * (x_max - x_min + 1e-9)
    y_pad = 0.04 * (y_max - y_min + 1e-9)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)

    # optional title (comment out if you want the same style as your example)
    # ax.set_title("UMAP of Image Embeddings")

    # ---- custom L indicator (labels INSIDE) ----
    origin = (0.06, 0.08)   # L corner, axes fraction
    length = 0.20
    lw = 3.8

    # L lines
    ax.plot([origin[0], origin[0]],
            [origin[1], origin[1] + length],
            transform=ax.transAxes, color="black", lw=lw,
            solid_capstyle="butt", clip_on=False)
    ax.plot([origin[0], origin[0] + length],
            [origin[1], origin[1]],
            transform=ax.transAxes, color="black", lw=lw,
            solid_capstyle="butt", clip_on=False)

    # INSIDE text placement:
    ax.text(origin[0] + 0.03,
            origin[1] + length / 2 + 0.03,
            "UMAP2",
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=20,
            color="black")

    ax.text(origin[0] + length / 2 + 0.03,
            origin[1] + 0.03,
            "UMAP1",
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=20,
            color="black")

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0],
               marker=markers[lbl], linestyle="",
               markerfacecolor=palette[lbl],
               markeredgecolor="white",
               markeredgewidth=1.0,
               markersize=18,
               label=lbl)
        for lbl in unique_labels
    ]
    # ax.legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="0.85", loc="upper right", fontsize=24)
    ax.legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="0.85", loc="lower right", fontsize=24)

    fig.tight_layout()

    output_path = output_dir / "umap_embedding.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"UMAP plot saved to: {output_path}")

    # save to svg
    output_path_svg = output_dir / "umap_embedding.svg"
    fig.savefig(output_path_svg, dpi=800, bbox_inches="tight", pad_inches=0, transparent=True)
    print(f"UMAP plot saved to: {output_path_svg}")

    plt.close(fig)

if __name__ == "__main__":
    main()
