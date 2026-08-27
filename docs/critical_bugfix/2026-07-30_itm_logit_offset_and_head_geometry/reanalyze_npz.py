#!/usr/bin/env python
"""npz 원자료 재분석 — 모델 재실행 없이 README §3.2~3.5 표를 재생성한다.

  (A) offset 비율의 꼬리 (p50/p90/p99/max) + 1-ρ (포화되지 않은 스케일)
  (B) 난이도 층화 — 난이도는 cos 기준 정답 순위(ITM과 독립인 외생 변수)
  (C) 불일치(instability) vs 결과(consequence) 분리 + R@1 McNemar 검정

실행:  python critical_bugfix/2026-07-30_itm_logit_offset_and_head_geometry/reanalyze_npz.py
"""
import os

import numpy as np
from scipy.stats import binomtest

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npz")
TAGS = ["teacher_blip_large", "baseline_ep19", "baseline_ep15", "schemeA_ep15"]


def load(tag, d):
    z = np.load(os.path.join(D, f"{tag}_{d}.npz"))
    return (z["z0"].astype(np.float64), z["z1"].astype(np.float64),
            z["cos"].astype(np.float64), z["is_gt"])


def rank_gt(S, GT):
    """점수 S 내림차순에서 정답의 0-based 순위 (없으면 후보 수)."""
    o = np.argsort(-S, axis=1, kind="stable")
    return np.array([(np.where(GT[i][o[i]])[0][0] if GT[i].any() else S.shape[1])
                     for i in range(len(S))])


def rowcorr(A, B):
    Ac, Bc = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    return (Ac * Bc).sum(1) / np.sqrt((Ac ** 2).sum(1) * (Bc ** 2).sum(1))


for d in ["i2t", "t2i"]:
    print("=" * 124)
    print(f"== {d} ==\n")
    print("(A) offset 비율의 꼬리 + 포화되지 않은 스케일")
    print(f"{'model':20s}{'1-ρ(z1,gap)':>13s}{'σm/σg mean':>12s}{'p50':>8s}{'p90':>8s}{'p99':>8s}{'max':>8s}"
          f"{'중심점 m':>11s}{'σ(m)':>8s}")
    for tag in TAGS:
        Z0, Z1, COS, GT = load(tag, d)
        GAP, M = Z1 - Z0, 0.5 * (Z1 + Z0)
        r = M.std(1, ddof=1) / GAP.std(1, ddof=1)
        rho = rowcorr(Z1, GAP)
        print(f"{tag:20s}{1-np.median(rho):13.6f}{r.mean():12.4f}{np.percentile(r,50):8.4f}"
              f"{np.percentile(r,90):8.4f}{np.percentile(r,99):8.4f}{r.max():8.4f}"
              f"{M.mean():11.4f}{M.std(1, ddof=1).mean():8.4f}")

    print(f"\n(B) 난이도 층화 (난이도 = cos 기준 정답 순위)")
    print(f"{'model':20s}{'bucket':>10s}{'n':>6s}{'σ(m)절대':>10s}{'σ(gap)':>9s}{'σm/σg':>8s}"
          f"{'argmax일치':>11s}{'정답순위변동':>13s}")
    for tag in TAGS:
        Z0, Z1, COS, GT = load(tag, d)
        GAP, M = Z1 - Z0, 0.5 * (Z1 + Z0)
        sm, sg = M.std(1, ddof=1), GAP.std(1, ddof=1)
        cr = rank_gt(COS, GT)
        r_z1, r_gap = rank_gt(Z1, GT), rank_gt(GAP, GT)
        am = Z1.argmax(1) == GAP.argmax(1)
        for nm, mk in [("easy(0)", cr == 0), ("mid(1-4)", (cr >= 1) & (cr < 5)), ("hard(5+)", cr >= 5)]:
            if mk.sum() == 0:
                continue
            print(f"{tag if nm == 'easy(0)' else '':20s}{nm:>10s}{int(mk.sum()):6d}"
                  f"{np.median(sm[mk]):10.4f}{np.median(sg[mk]):9.3f}{np.median(r := sm[mk]/sg[mk]):8.4f}"
                  f"{100*am[mk].mean():10.1f}%{100*(r_z1[mk] != r_gap[mk]).mean():12.1f}%")

    print(f"\n(C) 불일치 vs 결과 + R@1 McNemar")
    print(f"{'model':20s}{'argmax불일치':>13s}{'정답순위변동':>13s}{'R@1(z1)':>9s}{'R@1(gap)':>10s}{'ΔR@1':>8s}"
          f"{'b':>5s}{'c':>5s}{'McNemar p':>11s}")
    for tag in TAGS:
        Z0, Z1, COS, GT = load(tag, d)
        GAP = Z1 - Z0
        r_z1, r_gap = rank_gt(Z1, GT), rank_gt(GAP, GT)
        h1, hg = r_z1 < 1, r_gap < 1
        b, c = int((h1 & ~hg).sum()), int((hg & ~h1).sum())
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        print(f"{tag:20s}{100*(Z1.argmax(1) != GAP.argmax(1)).mean():12.1f}%"
              f"{100*(r_z1 != r_gap).mean():12.1f}%{100*h1.mean():9.1f}{100*hg.mean():10.1f}"
              f"{100*(h1.mean()-hg.mean()):+8.1f}{b:5d}{c:5d}{pv:11.3f}")
    print()
