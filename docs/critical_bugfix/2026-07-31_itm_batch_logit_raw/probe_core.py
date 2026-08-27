"""ITM 배치 로짓 프로브의 순수 함수 — 모델·데이터셋을 모른다.

블록 규약 (models/blip_pretrain.py:471-489 와 동일):
    [0  : B ]  pos      (img_i,     txt_i)
    [B  : 2B]  neg_img  (neg_img_i, txt_i)      ← 텍스트가 원본
    [2B : 3B]  neg_txt  (img_i,     neg_txt_i)  ← 이미지가 원본
"""
import csv

import numpy as np
import torch
import torch.nn.functional as F

BLOCK_NAMES = ("pos", "neg_img", "neg_txt")
COLUMNS = ("z1_nomatch", "z2_match", "m", "gap")


def neg_weights(sim):
    """in-batch negative 후보 가중치. blip_pretrain.py:448-451 과 동일.

    sim: [B, B] (이미 logit_scale 이 곱해진 값). 입력을 변형하지 않는다.
    """
    w = F.softmax(sim, dim=1) + 1e-4
    w = w.clone()
    w.fill_diagonal_(0)
    return w


def sample_neg_idx(weights):
    """행마다 multinomial 1개. 전역 RNG 를 쓰므로 torch.manual_seed 로 재현된다."""
    bs = weights.size(0)
    return torch.tensor(
        [int(torch.multinomial(weights[b], 1).item()) for b in range(bs)],
        dtype=torch.long,
    )


def split_blocks(logits):
    """[3B, 2] → {"pos": [B,2], "neg_img": [B,2], "neg_txt": [B,2]}"""
    n = logits.size(0)
    if n % 3 != 0:
        raise ValueError(f"logits rows must be a multiple of 3, got {n}")
    b = n // 3
    return {
        "pos": logits[0:b],
        "neg_img": logits[b : 2 * b],
        "neg_txt": logits[2 * b : 3 * b],
    }


def make_frame(batch_ids, row_ids, logits):
    """logits [N,2] → 열 dict. z1=class 0, z2=class 1."""
    arr = np.asarray(logits, dtype=np.float64)
    z1 = arr[:, 0]
    z2 = arr[:, 1]
    return {
        "batch": np.asarray(batch_ids, dtype=np.int64),
        "i": np.asarray(row_ids, dtype=np.int64),
        "z1_nomatch": z1,
        "z2_match": z2,
        "m": 0.5 * (z1 + z2),
        "gap": z2 - z1,
    }


def column_stats(frame):
    out = {}
    for c in COLUMNS:
        a = np.asarray(frame[c], dtype=np.float64)
        out[c] = {
            "mean": float(a.mean()),
            "var": float(a.var(ddof=1)) if a.size > 1 else float("nan"),
            "std": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        }
    return out


def write_block_csv(path, frame):
    st = column_stats(frame)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "i", *COLUMNS])
        for r in range(len(frame["batch"])):
            w.writerow(
                [int(frame["batch"][r]), int(frame["i"][r])]
                + [f"{float(frame[c][r]):.6f}" for c in COLUMNS]
            )
        for key in ("mean", "var", "std"):
            w.writerow([key, ""] + [f"{st[c][key]:.6f}" for c in COLUMNS])


def _softmax_mass(subset, whole):
    """sum(exp(subset)) / sum(exp(whole)) — max 를 빼서 오버플로를 막는다."""
    mx = float(np.max(whole))
    return float(np.exp(subset - mx).sum() / np.exp(whole - mx).sum())


def _p_pos_closed_form(delta):
    """e^d / (e^d + 2) 의 오버플로 안전 형태.

    d >= 0 에서는 1 / (1 + 2e^-d) 로 갈아탄다 (e^d 가 터지는 쪽을 피한다).
    d < 0 에서는 원식 그대로가 안전하다 (e^d -> 0).
    """
    d = float(delta)
    if d >= 0.0:
        return float(1.0 / (1.0 + 2.0 * np.exp(-d)))
    e = np.exp(d)
    return float(e / (e + 2.0))


def summarize_blocks(frames):
    """스펙 §8 의 판정 지표. frames 는 BLOCK_NAMES 를 키로 갖는 make_frame 결과."""
    pos_gap = np.asarray(frames["pos"]["gap"], dtype=np.float64)
    neg_gap = np.concatenate(
        [np.asarray(frames["neg_img"]["gap"], dtype=np.float64),
         np.asarray(frames["neg_txt"]["gap"], dtype=np.float64)]
    )
    all_gap = np.concatenate([pos_gap, neg_gap])
    delta = float(pos_gap.mean() - neg_gap.mean())

    out = {
        "delta": delta,
        "p_pos_merged_from_delta": _p_pos_closed_form(delta),
        "p_pos_merged_measured": _softmax_mass(pos_gap, all_gap),
        "n_per_block": {k: int(len(frames[k]["gap"])) for k in BLOCK_NAMES},
    }
    for name in BLOCK_NAMES:
        st = column_stats(frames[name])
        out[name] = {
            "stats": st,
            "sigma_m": st["m"]["std"],
            "sigma_gap": st["gap"]["std"],
        }
    return out
