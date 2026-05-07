import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Data ---
run1 = {
    "Adjusted Rand Index": 0.3838,
    "V-measure": 0.6147,
    "Fowlkes-Mallows Index": 0.4639,
    "Hungarian Accuracy": 0.4246,
    "B3 Precision/Recall/F1": (0.6835, 0.3581, 0.4700),
}
run2 = {
    "Adjusted Rand Index": 0.4401,
    "V-measure": 0.6730,
    "Fowlkes-Mallows Index": 0.5164,
    "Hungarian Accuracy": 0.4968,
    "B3 Precision/Recall/F1": (0.7198, 0.4334, 0.5410),
}

metrics = [
    "Adjusted Rand Index",
    "V-measure",
    "Fowlkes-Mallows Index",
    "Hungarian Accuracy",
    "B³ Precision",
    "B³ Recall",
    "B³ F1",
]

b3_1 = run1["B3 Precision/Recall/F1"]
b3_2 = run2["B3 Precision/Recall/F1"]

df = pd.DataFrame(
    {
        "Experiment": [
            run1["Adjusted Rand Index"],
            run1["V-measure"],
            run1["Fowlkes-Mallows Index"],
            run1["Hungarian Accuracy"],
            b3_1[0], b3_1[1], b3_1[2],
        ],
        "Virtual staining": [
            run2["Adjusted Rand Index"],
            run2["V-measure"],
            run2["Fowlkes-Mallows Index"],
            run2["Hungarian Accuracy"],
            b3_2[0], b3_2[1], b3_2[2],
        ],
    },
    index=metrics,
)

pretty_labels = [
    "Adjusted Rand Index",
    "V-measure",
    "Fowlkes–Mallows",
    "Hungarian Accuracy",
    "B³ Precision",
    "B³ Recall",
    "B³ F1",
]

# --- Publish-friendly defaults ---
# plt.rcParams.update({
#     "font.size": 9,
#     "axes.titlesize": 10,
#     "axes.labelsize": 9,
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# --- Better colors (colorblind-safe) ---
c_exp = "#0072B2"   # blue
c_vs  = "#D55E00"   # orange

y = np.arange(len(metrics))
h = 0.34  # bar "thickness" for each group

# 横向图一般更“窄”，但需要更高一点的高度容纳标签
fig, ax = plt.subplots(figsize=(5, 4.0), dpi=300)

bars1 = ax.barh(
    y - h/2, df["Experiment"].values, height=h,
    label="Experiment", color=c_exp,
    edgecolor="black", linewidth=0.4
)
bars2 = ax.barh(
    y + h/2, df["Virtual staining"].values, height=h,
    label="Virtual staining", color=c_vs,
    edgecolor="black", linewidth=0.4
)

# ax.set_xlabel("Score")
ax.set_title("KMeans Metric Comparison", fontsize=15)
ax.set_yticks(y)
ax.set_yticklabels(pretty_labels, rotation=45, fontsize=12)

ax.invert_yaxis()

ax.set_xlim(0, 1.05)
ax.grid(axis="x", alpha=0.25, linewidth=0.8)
ax.set_axisbelow(True)

ax.tick_params(axis='x', labelsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=12)

# 数值标签（写在条形右侧）
def label_bars_h(bars, pad=0.012):
    for b in bars:
        w = b.get_width()
        ax.text(
            w + pad,
            b.get_y() + b.get_height()/2,
            f"{w:.3f}",
            ha="left", va="center",
            fontsize=12,
            clip_on=False
        )

label_bars_h(bars1)
label_bars_h(bars2)

# 给右边留空，否则 Δ 可能被裁掉
ax.set_xlim(0, 1.2)

# no layout padding needed for horizontal
# save to svg

fig.patch.set_alpha(0)   # figure 背景透明
ax.patch.set_alpha(0)    # axes 背景透明（坐标轴区域）

fig.savefig(
    "./kmeans_metrics_comparison_publish_horizontal.svg",
    bbox_inches="tight",
    pad_inches=0,
    transparent=True
)
print("Saved figure to kmeans_metrics_comparison_publish_horizontal.svg")
plt.close(fig)
