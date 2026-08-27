"""ITM 양방향 KD 설계 근거 — 참여비(PR)·테일·온도 유도의 실측 검증.

프로브(2026-07-31) 로짓으로:
  1. 블록별 실측 PR(z1,z2) vs 가우시안 예측 B/e
  2. 블록 내 표준화 로짓의 excess kurtosis / skew  (테일 무게 판정)
  3. 참조분포(균등/가우시안/라플라스/t3)의 PR (B=40) — "PR↑⟺테일 가벼움" 시연
  4. PR = B·e^{-1/c²} 의 c 스윕 표
결과를 콘솔 + pr_tail_analysis.xlsx 로 저장.
"""
import os, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "2026-07-31_itm_batch_logit_raw", "csv")

def softmax(a):
    a = a - a.max(); e = np.exp(a); return e / e.sum()

def pr(vals, c=1.0):
    s = vals.std(ddof=0)
    if s < 1e-9: return float(len(vals))
    p = softmax(vals / (c * s))
    return 1.0 / np.sum(p ** 2)

def moments(x):
    x = np.asarray(x, float); m = x.mean(); s = x.std(ddof=0)
    z = (x - m) / s
    return z.mean()*0 + (z**3).mean(), (z**4).mean() - 3.0  # skew, excess kurt

# ---- 1+2. 블록별 실측 PR + 표준화 로짓 첨도/왜도 ----
rows = []
for f in sorted(glob.glob(os.path.join(CSV, "*.csv"))):
    df = pd.read_csv(f)
    df = df[pd.to_numeric(df["batch"], errors="coerce").notna()].copy()
    df["batch"] = df["batch"].astype(float).astype(int)
    name = os.path.basename(f).replace(".csv", "")
    rec = {"block": name}
    for col, tag in [("z1_nomatch", "z1"), ("z2_match", "z2")]:
        prs, pooledz = [], []
        for _, g in df.groupby("batch"):
            v = g[col].to_numpy(float)
            if len(v) < 5: continue
            prs.append(pr(v))
            pooledz.append((v - v.mean()) / (v.std(ddof=0) + 1e-9))  # per-batch 표준화 후 pool
        pooled = np.concatenate(pooledz)
        sk, ek = moments(pooled)
        rec[f"PR_{tag}"] = np.mean(prs)
        rec[f"kurt_{tag}"] = ek
        rec[f"skew_{tag}"] = sk
    rec["B"] = df.groupby("batch").size().mean()
    rec["PR_model_B/e"] = rec["B"] / np.e
    rows.append(rec)
per_block = pd.DataFrame(rows)

# ---- 3. 참조분포 PR (B=40) : 테일 무게 스펙트럼 ----
rng = np.random.default_rng(0); B, N = 40, 20000
def mc_pr(sampler):
    vals = [pr(sampler(B)) for _ in range(N)]; return float(np.mean(vals))
refs = {
    "uniform (가벼움, ek<0)":  lambda B: rng.uniform(-np.sqrt(3), np.sqrt(3), B),
    "gaussian (기준, ek=0)":   lambda B: rng.standard_normal(B),
    "laplace (무거움, ek=3)":  lambda B: rng.laplace(0, 1/np.sqrt(2), B),
    "student-t df=3 (매우무거움)": lambda B: rng.standard_t(3, B) / np.sqrt(3.0),
}
mc = pd.DataFrame([{"distribution": k, "PR (B=40)": mc_pr(v)} for k, v in refs.items()])

# ---- 4. PR = B·e^{-1/c²} c 스윕 ----
cs = [0.3, 0.5, 0.8, 1.0, 1.3, 2.0, 3.0]
csweep = pd.DataFrame({"c": cs, "PR = B·e^(-1/c^2), B=40": [40 * np.exp(-1/c**2) for c in cs]})

print("=== 1+2. 블록별 실측 PR + 표준화 로짓 테일 ===")
print(per_block.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
print("\n=== 3. 참조분포 PR (B=40) — 테일 무거울수록 PR 낮아짐 ===")
print(mc.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
print("\n=== 4. PR = B·e^(-1/c^2) 스윕 ===")
print(csweep.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

out = os.path.join(HERE, "pr_tail_analysis.xlsx")
with pd.ExcelWriter(out) as xl:
    per_block.to_excel(xl, sheet_name="per_block_PR_tail", index=False)
    mc.to_excel(xl, sheet_name="reference_PR", index=False)
    csweep.to_excel(xl, sheet_name="PR_vs_c", index=False)
print("\nwrote", out)
