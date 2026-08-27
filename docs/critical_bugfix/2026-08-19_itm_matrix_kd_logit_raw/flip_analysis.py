#!/usr/bin/env python
"""m 진동이 정답↔하드네거티브 순위를 실제로 뒤집는 비율.

리랭킹 = z2 = m + gap/2 로 후보 줄세움. anchor 마다 정답 vs 하드네거티브 한 쌍:
  i2t: 이미지 i 의 {정답텍스트(pos), neg텍스트(neg_txt)}
  t2i: 텍스트 i 의 {정답이미지(pos), neg이미지(neg_img)}
Δgap = gap_pos - gap_neg,  Δm = m_pos - m_neg.
  gap 판정: 정답승 iff Δgap>0.   z2 판정: 정답승 iff Δgap/2 + Δm > 0.
  flip = 두 판정 불일치 (= m 때문에 답 바뀜).  harmful = gap선 정답승인데 z2선 정답패.
정렬 CSV(같은 3B 쌍) 산술만. z2 단독 기준이라 실제 리트리벌(z2+sim)의 flip 상한.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "aligned", "csv")
MODELS = ("teacher", "student_baseline", "student_itm_kxk")
# (방향, 정답블록, 네거티브블록)
DIRS = (("i2t", "pos", "neg_txt"), ("t2i", "pos", "neg_img"))


def load(tag, block):
    df = pd.read_csv(os.path.join(CSV, f"{tag}_{block}.csv"))
    return df[pd.to_numeric(df["batch"], errors="coerce").notna()].copy().reset_index(drop=True)


def stats(model, direction, pos_b, neg_b):
    p, n = load(model, pos_b), load(model, neg_b)
    assert (p[["batch", "i"]].to_numpy() == n[["batch", "i"]].to_numpy()).all(), "anchor 정렬 깨짐"
    dgap = p["gap"].to_numpy(float) - n["gap"].to_numpy(float)
    dm = p["m"].to_numpy(float) - n["m"].to_numpy(float)
    gap_win = dgap > 0
    z2_win = (dgap / 2 + dm) > 0
    flip = gap_win != z2_win
    harmful = gap_win & ~z2_win            # 정답이 gap선 이기는데 m 때문에 짐
    ratio = np.abs(dm) / (np.abs(dgap) / 2 + 1e-9)
    # close-gap: |Δgap| 하위 25% 에서의 flip
    q25 = np.quantile(np.abs(dgap), 0.25)
    close = np.abs(dgap) <= q25
    return {
        "model": model, "direction": direction, "n": len(p),
        "flip_rate": flip.mean(), "harmful_rate": harmful.mean(),
        "frac_|dm|>|dgap|/2": (np.abs(dm) > np.abs(dgap) / 2).mean(),
        "median_ratio": np.median(ratio),
        "flip_rate_closegap25%": flip[close].mean(),
    }


def main():
    rows = [stats(m, d, pb, nb) for m in MODELS for (d, pb, nb) in DIRS]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== m-induced rank flip (정답 vs 하드네거티브) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    with pd.ExcelWriter(os.path.join(HERE, "aligned", "flip_analysis.xlsx")) as xl:
        df.to_excel(xl, sheet_name="flip", index=False)

    # 그래프: flip_rate & harmful_rate (모델×방향)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(MODELS)); w = 0.2
    col = {"teacher": "#d62728", "student_baseline": "#1f77b4", "student_itm_kxk": "#2ca02c"}
    for di, (d, _, _) in enumerate(DIRS):
        fr = [df[(df.model == m) & (df.direction == d)]["flip_rate"].iloc[0] for m in MODELS]
        hr = [df[(df.model == m) & (df.direction == d)]["harmful_rate"].iloc[0] for m in MODELS]
        ax.bar(x + (di * 2 - 1) * w, fr, w, color=[col[m] for m in MODELS],
               alpha=1.0 if di == 0 else 0.55, edgecolor="k", label=f"flip {d}")
        ax.bar(x + (di * 2) * w, hr, w, color=[col[m] for m in MODELS],
               alpha=1.0 if di == 0 else 0.55, edgecolor="k", hatch="//", label=f"harmful {d}")
    ax.set_xticks(x + w / 2); ax.set_xticklabels(MODELS)
    ax.set_ylabel("rate"); ax.set_title("m-induced flip rate (pos vs hard-neg, z2-only upper bound)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "aligned", "flip_analysis.png"), dpi=120)
    print("\nwrote flip_analysis.xlsx, flip_analysis.png")


if __name__ == "__main__":
    main()
