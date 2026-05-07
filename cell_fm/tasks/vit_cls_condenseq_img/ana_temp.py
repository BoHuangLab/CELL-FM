import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================
# 1. 基本设置：AA 顺序 & 物理先验分数
# ============================

AA = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C',
      'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

sticker_score = {
    'R': 0.5,
    'H': 0.25,
    'K': 0.5,
    'D': -1.0,
    'E': -1.0,
    'S': 0.25,
    'T': 0.25,
    'N': 1.0,
    'Q': 1.0,
    'C': -0.5,
    'G': 0.0,
    'P': -1.5,
    'A': 0.25,
    'V': 0.5,
    'I': 0.5,
    'L': 0.5,
    'M': 0.5,
    'F': 1.5,
    'Y': 1.5,
    'W': 1.5,
}


def build_theoretical_matrix(AA, sticker_score):
    """
    构建 20x20 的物理先验突变矩阵：
    mut_matrix[i,j] = s(AA_j) - s(AA_i)
    """
    n = len(AA)
    mut_matrix = np.zeros((n, n), dtype=float)
    for i, a1 in enumerate(AA):
        for j, a2 in enumerate(AA):
            mut_matrix[i, j] = sticker_score[a2] - sticker_score[a1]
    return mut_matrix


def flatten_valid_entries(empirical, theoretical, exclude_diagonal=True):
    """
    展平矩阵，只保留两边都不是 NaN 的元素。
    可选：去掉对角线。
    """
    assert empirical.shape == theoretical.shape
    n = empirical.shape[0]

    mask = np.isfinite(empirical) & np.isfinite(theoretical)
    if exclude_diagonal:
        eye = np.eye(n, dtype=bool)
        mask = mask & (~eye)

    emp_flat = empirical[mask]
    theo_flat = theoretical[mask]
    return emp_flat, theo_flat, mask


def pearson_corr(x, y):
    """
    简单算个 Pearson 相关
    """
    if len(x) < 2:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def sign_agreement(x, y, zero_tol=1e-8):
    """
    计算符号一致率：
    - 先根据 zero_tol 把非常接近 0 的视作 0
    - 然后统计 sign(x) == sign(y) 的比例
    """
    x_sign = np.zeros_like(x)
    y_sign = np.zeros_like(y)

    x_sign[x > zero_tol] = 1
    x_sign[x < -zero_tol] = -1

    y_sign[y > zero_tol] = 1
    y_sign[y < -zero_tol] = -1

    same = (x_sign == y_sign)
    return same.mean(), x_sign, y_sign


def main():
    out_dir = Path(".")
    out_dir.mkdir(exist_ok=True, parents=True)

    # ============================
    # 2. 读取你的实验/模型突变矩阵
    # ============================
    # 这里假设你已经 np.save('mutation_median_matrix.npy', median_mat)
    empirical = np.load("mutation_median_matrix.npy")  # shape: (20, 20)

    # ============================
    # 3. 构建理论突变矩阵
    # ============================
    theoretical = build_theoretical_matrix(AA, sticker_score)

    # ============================
    # 4. 展平 & 相关性分析
    # ============================
    emp_flat, theo_flat, mask = flatten_valid_entries(empirical, theoretical,
                                                      exclude_diagonal=True)

    r = pearson_corr(emp_flat, theo_flat)
    sign_acc, emp_sign, theo_sign = sign_agreement(emp_flat, theo_flat)

    print(f"Number of valid mutation pairs: {len(emp_flat)}")
    print(f"Pearson correlation (theory vs empirical): {r:.3f}")
    print(f"Sign agreement ratio: {sign_acc*100:.1f}%")

    # ============================
    # 5. 画散点图：理论预测 vs 实验矩阵
    # ============================
    plt.figure(figsize=(6, 6))
    plt.scatter(theo_flat, emp_flat, alpha=0.5, s=10)
    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(0, color='gray', linewidth=1)
    plt.xlabel("Theoretical Δscore (sticker_score[A2] - sticker_score[A1])")
    plt.ylabel("Empirical median Δ (mt_pred - wt_pred)")
    plt.title(f"Theory vs empirical mutation effect\nPearson r = {r:.3f}, sign acc = {sign_acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig(out_dir / "theory_vs_empirical_scatter.png", dpi=300)
    plt.close()

    # ============================
    # 6. (可选) 行/列层面的相关性
    #    看哪些 from / to 氨基酸行为更符合理论
    # ============================
    n = empirical.shape[0]
    row_corrs = []
    col_corrs = []

    for i in range(n):
        # row: 固定 from = AA[i]，看所有 to
        row_mask = mask[i, :]
        emp_row = empirical[i, :][row_mask]
        theo_row = theoretical[i, :][row_mask]
        row_corrs.append(pearson_corr(emp_row, theo_row))

        # column: 固定 to = AA[i]，看所有 from
        col_mask = mask[:, i]
        emp_col = empirical[:, i][col_mask]
        theo_col = theoretical[:, i][col_mask]
        col_corrs.append(pearson_corr(emp_col, theo_col))

    row_corrs = np.array(row_corrs)
    col_corrs = np.array(col_corrs)

    print("\nPer-from-AA row-wise correlation (theory vs empirical):")
    for aa, c in zip(AA, row_corrs):
        print(f"  from {aa}: r = {c:.3f}")

    print("\nPer-to-AA column-wise correlation (theory vs empirical):")
    for aa, c in zip(AA, col_corrs):
        print(f"  to   {aa}: r = {c:.3f}")

    # 也可以简单画出这两个条形图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(n), row_corrs)
    plt.xticks(range(n), AA, rotation=90)
    plt.ylabel("Pearson r")
    plt.title("Row-wise corr (fixed FROM AA)")

    plt.subplot(1, 2, 2)
    plt.bar(range(n), col_corrs)
    plt.xticks(range(n), AA, rotation=90)
    plt.title("Column-wise corr (fixed TO AA)")

    plt.tight_layout()
    plt.savefig(out_dir / "row_col_correlation.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
