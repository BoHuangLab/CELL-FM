import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
import os

def parse_mutations(name: str):
    """
    从字符串中提取突变，支持形如 F3S, A10V, Y66F 等，并支持下划线分隔。
    返回列表：[(pos0_based, from_aa, to_aa), ...]
    """
    muts = re.findall(r'([A-Z])(\d{1,3})([A-Z])', name)
    parsed = []
    for from_aa, pos, to_aa in muts:
        pos = int(pos)
        if 1 <= pos <= 66:
            parsed.append((pos - 1, from_aa, to_aa))
    return parsed

def infer_substitution_label(proteins):
    """
    推断颜色代表的替换类型（例如 F→S）。
    如果只有一种 (from,to)，就返回那一种；否则返回 'multiple substitutions'。
    """
    pairs = set()
    for p in proteins:
        for _, a, b in parse_mutations(p):
            pairs.add((a, b))
    if len(pairs) == 1:
        a, b = next(iter(pairs))
        return f"{a}→{b}"
    elif len(pairs) == 0:
        return "no mutations"
    else:
        # 如果真的出现多种替换，可用这个兜底（你说都是一样的，通常不会走到这里）
        return "multiple substitutions"

def plot_mutation_blocks(
    proteins,
    show_names,
    length=66,
    wt_color="#BDBDBD",
    mut_color="#FF7043",
    edge_color="#FFFFFF",
    figsize_per_row=0.45,
    out_path="mutation_blocks.png",
    cell_w=0.55,
    cell_h=1.0,
    legend=True
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle

    if len(proteins) != len(show_names):
        raise ValueError(f"len(proteins)={len(proteins)} must equal len(show_names)={len(show_names)}")

    n = len(proteins)
    total_w = length * cell_w
    total_h = n * cell_h

    fig_h = max(2.5, n * figsize_per_row)
    fig_w = max(6, total_w * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.set_aspect('auto')

    ax.set_yticks([(i + 0.5) * cell_h for i in range(n)])
    ax.set_yticklabels(show_names, fontsize=9)

    xticks_pos = list(range(0, length + 1, 5))
    ax.set_xticks([x * cell_w for x in xticks_pos])
    ax.set_xticklabels([str(x) for x in xticks_pos], fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # 绘制方块
    for row, name in enumerate(proteins):
        muts = parse_mutations(name)
        mut_map = {pos: (from_aa, to_aa) for (pos, from_aa, to_aa) in muts}

        y0 = row * cell_h
        for col in range(length):
            x0 = col * cell_w
            color = mut_color if col in mut_map else wt_color

            rect = patches.Rectangle(
                (x0, y0), cell_w, cell_h,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=0.6
            )
            ax.add_patch(rect)

    # 去掉 title
    ax.set_title("")

    # 用 legend 表明颜色含义与 mutation 类型（并让 legend 里的色块形状与 cell 一致）
    if legend:
        sub_label = infer_substitution_label(proteins)  # e.g. F→S

        # ✅ 用 Rectangle 作为 legend handle，并用 handlelength/handleheight 调成与 cell_w:cell_h 一致
        mut_handle = Rectangle(
            (0, 0), 1, 1,  # 实际大小由 legend 的 handlelength/handleheight 控制
            facecolor=mut_color,
            edgecolor=edge_color,
            linewidth=0.6,
            label=f"Mutation ({sub_label})"
        )

        # 给顶部留空间放 legend（尤其你 bbox_inches="tight"）
        # plt.tight_layout(rect=[0, 0, 1, 0.92])

        handle_h = 1.2
        handle_l = 0.4

        leg = ax.legend(
            handles=[mut_handle],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=1,
            frameon=True,          # ✅ 打开外框
            fancybox=True,        # ✅ 方角矩形（True 会变圆角）
            framealpha=1.0,        # ✅ 不透明
            fontsize=10,
            handleheight=handle_h,
            handlelength=handle_l,
            borderpad=0.3,
            labelspacing=0.4
        )
    else:
        plt.tight_layout()

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    protein_name = "NUP98"
    all_csv_paths = sorted(glob(f"/hpc/reference/opencell/condenseq/seq2img_single/{protein_name}/protein_*/pred_vs_intensity.csv"))

    proteins = []
    for csv_path in all_csv_paths:
        folder_name = os.path.basename(os.path.dirname(csv_path))
        variant_name = folder_name.split("_", 1)[-1]
        proteins.append(variant_name)

    show_names = []
    for i, protein in enumerate(proteins):
        if protein == protein_name:
            show_names.append(f"{protein_name}_WT")
        else:
            show_names.append(f"{protein_name}_M{i}")
    print(show_names)

    plot_mutation_blocks(
        proteins, show_names,
        length=66,
        out_path=f"{protein_name}_mut_blocks.svg",
        figsize_per_row=0.1,
        cell_w=0.1, 
        cell_h=1.0   # 你想更高就改大点，比如 1.2；想矮点就改小，比如 0.8
    )