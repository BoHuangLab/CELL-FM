import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def moving_average(y, window):
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def binned_rate(x, y, n_bins=20, min_count=20):
    """Return bin centers and empirical P(y=1) per bin."""
    x = np.asarray(x)
    y = np.asarray(y)

    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 3:
        return np.array([]), np.array([]), np.array([])

    bin_id = np.digitize(x, bins[1:-1], right=False)
    centers, rates, counts = [], [], []

    for b in range(len(bins) - 1):
        m = bin_id == b
        c = int(m.sum())
        if c < min_count:
            continue
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        rates.append(float(y[m].mean()))
        counts.append(c)

    return np.asarray(centers), np.asarray(rates), np.asarray(counts)


def plot_pred_vs_intensity(
    x, y, title, save_path,
    window_size=32,
    n_bins=24,
    min_bin_count=30,
    dpi=220
):
    x = np.asarray(x)
    y = np.asarray(y).astype(float)

    # sort by x for smooth curves
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # moving average as estimated probability
    y_ma = moving_average(y, window_size)

    # binned empirical rate
    xc, yr, counts = binned_rate(x, y, n_bins=n_bins, min_count=min_bin_count)

    # jitter binary points for visibility
    rng = np.random.default_rng(0)
    y_jit = y + rng.normal(0, 0.03, size=len(y))
    y_jit = np.clip(y_jit, -0.05, 1.05)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # raw points
    ax.scatter(
        x, y_jit,
        s=8, alpha=0.15, linewidths=0,
        label="single cells (jittered)"
    )

    # binned rate (cleaner summary)
    if len(xc) > 0:
        ax.plot(
            xc, yr,
            marker="o", markersize=4, linewidth=1.5,
            label=f"empirical rate (binned, ≥{min_bin_count}/bin)"
        )

    # moving average curve
    ax.plot(
        x, y_ma,
        linewidth=2.2,
        label=f"moving average (window={window_size})"
    )

    # cosmetics
    ax.set_title(title, pad=10)
    ax.set_xlabel("Protein intensity level")
    ax.set_ylabel("P(condensate) / predicted class")
    ax.set_ylim(-0.05, 1.05)

    # make x nicer
    ax.margins(x=0.02)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


# =========================
# main loop
# =========================
data_path = Path("/hpc/reference/opencell/condenseq/seq2img_blind_screening_hp_idr")
meta_data = pd.read_csv(
    "/home/dihan.zheng/workspace/dihan.zheng/dataset/human_protein/"
    "human_idr_regions_plddt_only_chopped_unique_seq_with_names.csv"
)

window_size = 32

for _, row in tqdm(meta_data.iterrows(), total=len(meta_data)):
    protein_index = int(row["index"])
    csv_path = data_path / f"{protein_index:06d}" / "pred_vs_intensity.csv"
    pred_df = pd.read_csv(csv_path)

    x = pred_df["protein_intensity_level"].to_numpy()
    y = pred_df["predicted_class"].to_numpy()

    save_path = data_path / f"{protein_index:06d}" / "pred_vs_intensity_ma.png"
    title = f'Index: {protein_index} | {row.get("protein_name","")}'
    plot_pred_vs_intensity(
        x, y, title, save_path,
        window_size=window_size,
        n_bins=24,
        min_bin_count=30,
        dpi=220
    )