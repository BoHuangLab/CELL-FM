import pandas as pd
import numpy as np

# 读取文件
df = pd.read_csv("cell_fm/tasks/cell_fm_cs/all_mutants_kappa.csv")

# κ 列
kappa = df["kappa"].values

# 定义区间边界
bins = np.arange(0.1, 1.01, 0.1)   # 0.1,0.2,...,1.0
labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

# 将 κ 放入区间
df["kappa_bin"] = pd.cut(df["kappa"], bins=bins, labels=labels, include_lowest=True)

# 存储结果
selected = []

# 对每个 bin 随机选一条
for label in labels:
    subset = df[df["kappa_bin"] == label]
    if len(subset) > 0:
        chosen = subset.sample(1, random_state=42)  # 固定 seed
        selected.append(chosen)

# 合并
panel_df = pd.concat(selected).reset_index(drop=True)

# 保存结果
panel_df.to_csv("cell_fm/tasks/cell_fm_cs/kappa_binned_panel.csv", index=False)
print(panel_df[["kappa_bin", "kappa"]])
