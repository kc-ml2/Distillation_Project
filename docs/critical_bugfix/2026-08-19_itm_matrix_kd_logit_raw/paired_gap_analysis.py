#!/usr/bin/env python
"""행-정렬 CSV 로 paired gap 분석. 같은 3B 쌍(행 N 동일)이므로
Δ = teacher_gap - student_gap 를 쌍마다 직접 구한다.

raw std(Δ) : 스케일 차이 + 추종오차 둘 다 포함.
ρ          : Pearson 상관 (스케일 무관, "티처 순위를 따라가나").
resid std  : student=a·teacher+b 최소자승 후 잔차 std (스케일 맞춘 뒤 순수 추종오차).
baseline vs itm_kxk 비교 → 증류가 쌍 단위로 티처 gap 을 따라가게 만들었나.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "aligned", "csv")
BLOCKS = ("pos", "neg_img", "neg_txt")
STUDENTS = ("student_baseline", "student_itm_kxk")


def load(tag, block):
    df = pd.read_csv(os.path.join(CSV, f"{tag}_{block}.csv"))
    return df[pd.to_numeric(df["batch"], errors="coerce").notna()].copy().reset_index(drop=True)


def resid_std(x, y):
    """y = a x + b 최소자승 잔차 std (스케일·오프셋 맞춘 뒤 추종오차)."""
    a, b = np.polyfit(x, y, 1)
    return float((y - (a * x + b)).std(ddof=1)), float(a)


def analyze(col):
    rows = []
    # 블록별 + 전체(pooled)
    for block in list(BLOCKS) + ["ALL"]:
        if block == "ALL":
            t = np.concatenate([load("teacher", b)[col].to_numpy(float) for b in BLOCKS])
            svals = {s: np.concatenate([load(s, b)[col].to_numpy(float) for b in BLOCKS]) for s in STUDENTS}
        else:
            # 페어링 검산: batch,i 가 세 태그에서 동일해야 함
            ids = [load(tag, block)[["batch", "i"]].to_numpy() for tag in ("teacher", *STUDENTS)]
            assert (ids[0] == ids[1]).all() and (ids[0] == ids[2]).all(), f"pairing broken @ {block}"
            t = load("teacher", block)[col].to_numpy(float)
            svals = {s: load(s, block)[col].to_numpy(float) for s in STUDENTS}
        for s in STUDENTS:
            sv = svals[s]; d = t - sv
            rs, a = resid_std(t, sv)
            rows.append({"col": col, "block": block, "student": s, "n": len(t),
                         "std_teacher": t.std(ddof=1), "std_student": sv.std(ddof=1),
                         "mean_delta": d.mean(), "std_delta_raw": d.std(ddof=1),
                         "rho": np.corrcoef(t, sv)[0, 1], "slope": a, "resid_std": rs})
    return pd.DataFrame(rows)


def scatter(col, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, s in zip(axes, STUDENTS):
        t = np.concatenate([load("teacher", b)[col].to_numpy(float) for b in BLOCKS])
        sv = np.concatenate([load(s, b)[col].to_numpy(float) for b in BLOCKS])
        ax.scatter(t, sv, s=6, alpha=0.3, color="#2ca02c" if "kxk" in s else "#1f77b4")
        lim = [min(t.min(), sv.min()), max(t.max(), sv.max())]
        ax.plot(lim, lim, "k--", lw=1, label="y=x")
        a, b = np.polyfit(t, sv, 1); xs = np.array(lim)
        ax.plot(xs, a * xs + b, "r-", lw=1.2, label=f"fit slope={a:.2f}")
        rho = np.corrcoef(t, sv)[0, 1]; rs, _ = resid_std(t, sv)
        ax.set_title(f"{s}\nrho={rho:.3f}  std(Δ)={ (t-sv).std(ddof=1):.2f}  resid_std={rs:.2f}")
        ax.set_xlabel(f"teacher {col}"); ax.set_ylabel(f"student {col}"); ax.legend(loc="upper left")
    fig.suptitle(f"paired {col}: teacher vs student  (same 3B pairs)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "aligned", fname), dpi=120)
    print("wrote", fname)


def main():
    dfg = analyze("gap"); dfm = analyze("m")
    pd.set_option("display.width", 220)
    print("=== paired GAP ===\n", dfg.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\n=== paired M ===\n", dfm.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    with pd.ExcelWriter(os.path.join(HERE, "aligned", "paired_gap_analysis.xlsx")) as xl:
        dfg.to_excel(xl, sheet_name="paired_gap", index=False)
        dfm.to_excel(xl, sheet_name="paired_m", index=False)
    scatter("gap", "paired_gap_scatter.png")
    print("\nwrote paired_gap_analysis.xlsx, paired_gap_scatter.png")


if __name__ == "__main__":
    main()
