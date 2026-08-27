#!/usr/bin/env python
"""ITM 배치 로짓 분리도 3-way 시각화 + 엑셀.

teacher(BLIP-large) / baseline student(ep19, 무증류) / distilled student(itm_matrix_kd_configfix ep19)
세 모델의 블록별 m(중점)·gap(분리도=z2_match-z1_nomatch) 분포를 겹쳐 그린다.

색 = 모델, 선스타일 = 블록. 범례에 σ 표기 (예전 dist_m.png 관행 유지).
csv_distributions.xlsx : 블록×모델 전체 통계.  summary.xlsx : delta/P_pos/σ 요약.
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "aligned")                 # 산출물도 aligned/ 로 (misaligned 구버전과 분리)
ALIGNED_CSV = os.path.join(OUT, "csv")              # 행-정렬 3-모델 (probe_aligned.py 산출)
BLOCKS = ("pos", "neg_img", "neg_txt")

# (라벨, csv_dir, tag) — 전부 같은 시드·같은 3B 쌍 위에서 측정 (행 N 동일)
SERIES = [
    ("teacher",         ALIGNED_CSV, "teacher"),
    ("student_baseline", ALIGNED_CSV, "student_baseline"),
    ("student_itm_kxk",  ALIGNED_CSV, "student_itm_kxk"),
]
COLOR = {"teacher": "#d62728", "student_baseline": "#1f77b4", "student_itm_kxk": "#2ca02c"}
LS = {"pos": "-", "neg_img": "--", "neg_txt": ":"}


def load(csv_dir, tag, block):
    """batch 행만(요약 3행 제외) 남긴 DataFrame."""
    df = pd.read_csv(os.path.join(csv_dir, f"{tag}_{block}.csv"))
    return df[pd.to_numeric(df["batch"], errors="coerce").notna()].copy()


def plot_col(col, xlim, fname, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(-xlim, xlim, 81)
    for label, cdir, tag in SERIES:
        for block in BLOCKS:
            v = load(cdir, tag, block)[col].to_numpy(float)
            sig = v.std(ddof=1)
            ax.hist(v, bins=bins, histtype="step", color=COLOR[label], ls=LS[block],
                    lw=1.6, label=f"{label}_{block} (σ={sig:.3f})")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-xlim, xlim); ax.set_xlabel(col); ax.set_ylabel("count")
    ax.set_title(title)
    # 범례 2단: 모델(색) + 블록(선스타일)
    models = ("teacher", "student_baseline", "student_itm_kxk")
    model_h = [Line2D([0], [0], color=COLOR[m], lw=2) for m in models]
    block_h = [Line2D([0], [0], color="k", ls=LS[b], lw=1.6) for b in BLOCKS]
    leg1 = ax.legend(model_h, models, title="model (color)",
                     loc="upper left", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(block_h, BLOCKS, title="block (style)", loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=120)
    print("wrote", fname)


def stats_table():
    rows = []
    for label, cdir, tag in SERIES:
        for block in BLOCKS:
            df = load(cdir, tag, block)
            rec = {"model": label, "block": block, "n": len(df)}
            for c in ("z1_nomatch", "z2_match", "m", "gap"):
                a = df[c].to_numpy(float)
                rec[f"{c}_mean"] = a.mean(); rec[f"{c}_std"] = a.std(ddof=1)
            rows.append(rec)
    return pd.DataFrame(rows)


def summary_table():
    """모델별 delta = mean(gap_pos) - mean(gap_neg_pooled), P_pos(merged), σ(m)_pos."""
    rows = []
    for label, cdir, tag in SERIES:
        gap = {b: load(cdir, tag, b)["gap"].to_numpy(float) for b in BLOCKS}
        pos = gap["pos"]; neg = np.concatenate([gap["neg_img"], gap["neg_txt"]])
        allg = np.concatenate([pos, neg]); mx = allg.max()
        p_pos = np.exp(pos - mx).sum() / np.exp(allg - mx).sum()
        delta = pos.mean() - neg.mean()
        sm = load(cdir, tag, "pos")["m"].to_numpy(float).std(ddof=1)
        rows.append({"model": label, "delta_pos_vs_neg": delta,
                     "P_pos_if_merged": p_pos, "sigma_m_pos": sm,
                     "sigma_gap_pos": pos.std(ddof=1)})
    return pd.DataFrame(rows)


def main():
    plot_col("m", 0.245, "dist_m.png", "m = 0.5(z1+z2)  midpoint  (larger sigma = farther from teacher)  3-way")
    plot_col("gap", 12.0, "dist_gap.png", "gap = z2_match - z1_nomatch  (ITM separability)  3-way")
    dist = stats_table(); summ = summary_table()
    print("\n=== per-block stats ===\n", dist.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\n=== summary ===\n", summ.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    with pd.ExcelWriter(os.path.join(OUT, "csv_distributions.xlsx")) as xl:
        dist.to_excel(xl, sheet_name="per_block_stats", index=False)
    with pd.ExcelWriter(os.path.join(OUT, "summary.xlsx")) as xl:
        summ.to_excel(xl, sheet_name="summary", index=False)
    print("\nwrote csv_distributions.xlsx, summary.xlsx")


if __name__ == "__main__":
    main()
