import itertools
import math
import random
import csv
from typing import Iterator, List
from pathlib import Path

def generate_all_with_fixed_Y(n_repeats: int = 11) -> Iterator[str]:
    """
    原序列模块：[Y, GGG, NN] * n_repeats
    约束1：Y 的模块位点固定不动（位置 0,3,6,...）
    约束2：同一种模块连续最多 2 个（在该固定 Y 情况下会自动满足）
    生成：所有去重 shuffle 的最终“氨基酸序列字符串”
    """
    total_modules = 3 * n_repeats  # 33
    y_positions = list(range(0, total_modules, 3))  # 0,3,6,...,30
    free_positions = [i for i in range(total_modules) if i not in y_positions]  # 22 positions

    # 在 22 个 free_positions 里选 11 个放 GGG，其余放 NN
    for ggg_idx_set in itertools.combinations(range(len(free_positions)), n_repeats):
        modules = [None] * total_modules
        for p in y_positions:
            modules[p] = "Y"

        ggg_pos = {free_positions[i] for i in ggg_idx_set}
        for p in free_positions:
            modules[p] = "GGG" if p in ggg_pos else "NN"

        yield "".join(modules)

def count_all_with_fixed_Y(n_repeats: int = 11) -> int:
    # 22 选 11
    return math.comb(2 * n_repeats, n_repeats)

def random_sample_with_fixed_Y(
    n_repeats: int = 11,
    N: int = 10000,
    seed: int | None = 0,
) -> List[str]:
    """
    在所有 C(22,11) 个序列里“均匀”随机抽样 N 条（不重复）。
    方法：对 22 个自由位点随机选 11 个放 GGG，其余放 NN。
    由于总空间 705,432 不大，拒绝采样去重很快。
    """
    rng = random.Random(seed)

    total_modules = 3 * n_repeats
    y_positions = list(range(0, total_modules, 3))
    free_positions = [i for i in range(total_modules) if i not in y_positions]  # 22

    max_total = count_all_with_fixed_Y(n_repeats)
    if N > max_total:
        raise ValueError(f"N={N} exceeds total number of unique sequences {max_total}")

    sampled = set()
    while len(sampled) < N:
        ggg_pos = set(rng.sample(free_positions, n_repeats))  # 22 里选 11
        modules = [""] * total_modules
        for p in y_positions:
            modules[p] = "Y"
        for p in free_positions:
            modules[p] = "GGG" if p in ggg_pos else "NN"
        sampled.add("".join(modules))

    return list(sampled)

def save_to_csv(seqs: List[str], out_csv: str = "sampled_sequences.csv") -> None:
    """
    保存到 CSV：两列
    - id: 1..N
    - sequence: 拼接后的序列
    """
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence"])
        for i, s in enumerate(seqs, start=1):
            w.writerow([i, s])

if __name__ == "__main__":
    n_repeats = 11
    print("count =", count_all_with_fixed_Y(n_repeats))  # 705432

    # randomly sample N sequences (unique)
    N = 10000
    seqs = random_sample_with_fixed_Y(n_repeats=n_repeats, N=N, seed=0)

    # save to csv
    out_root = Path("cell_fm/tasks/cell_fm_cs/data/YGN")
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / "panel.csv"
    
    save_to_csv(seqs, out)
    print(f"saved {len(seqs)} sequences to {out}")

    # 示例：打印前 5 条
    for i, s in enumerate(seqs[:5]):
        print(i, s)
