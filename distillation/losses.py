import torch
import torch.nn.functional as F


def lm_distill_loss(student_logits, teacher_logits, decoder_targets, temp):
    """Token-level LM logit KD (teacher-forced).

    Args:
        student_logits: [B, L, V] student decoder logits, require grad.
        teacher_logits: [B, L, V] teacher decoder logits (constants).
        decoder_targets: [B, L] long, pad 위치는 -100 (BLIP LM CE와 동일 규칙).
        temp: 고정 증류 온도 T (config distill.lm.temp).

    Returns:
        Scalar = 유효 토큰당 평균 KL(teacher ‖ student) × T².
        CE와 동일한 shift 프레임: 위치 i 로짓이 토큰 i+1을 예측하므로
        마지막 위치를 버리고, targets를 한 칸 미뤄 유효 마스크를 만든다.
    """
    assert student_logits.shape[-1] == teacher_logits.shape[-1], ( # [B, L, V] 형태. 보캐블러리 사이즈를 뽑아서 보는거
        f"vocab size mismatch: student {student_logits.shape[-1]} "
        f"vs teacher {teacher_logits.shape[-1]}"
    )
    s = student_logits[:, :-1, :] # 마지막 토큰을 잘라냄. 칸 만드려고 [B, L-1, V]
    t = teacher_logits[:, :-1, :].detach() # 티처 쪽은 그래디언트가 흐르면 안되기 때문에 detach로 계산 그래프에서 끊어버림. 안전장치를 다는것
    valid = decoder_targets[:, 1:] != -100      # [B, L-1], 디코더 타겟을 L-1개만 true가 나오도록 하게
    # 이렇게 하면 지금 토큰이 맞춰야 할 위치 = 다음 토큰인 상태를 valid로 불리안 마스크 제작
    # 밑에처럼 인덱싱을 하면 벨리드 토큰 위치를 싹 읽으면서 일렬로 내보냄.
    s = s[valid]                                 # [N_valid, V] (true 위치들만 싹 모아서 일렬로 나열하기)
    t = t[valid]                                 # 이것도 true 위치들만 싹 나열하기
    return F.kl_div(
        F.log_softmax(s / temp, dim=-1), 
        F.softmax(t / temp, dim=-1),
        reduction="batchmean",                   # N_valid로 나눔 = 유효 토큰당 평균
    ) * (temp ** 2)
    
    # 이거 변경함 8.14
    # return F.kl_div( 
    #     F.log_softmax(s / temp, dim=-1), 
    #     F.log_softmax(t / temp, dim=-1),
    #     reduction="batchmean",                   # N_valid로 나눔 = 유효 토큰당 평균
    #     log_target=True,
    # ) * (temp ** 2)

# 이건 안쓰는놈같음
def itm_matrix_kd_loss(student_logits, teacher_logits, direction, temp):
    """Full B×B ITM 매치-로짓 매트릭스 관계 KD (ITC KD의 ITM 미러).

    student_logits, teacher_logits: [B, B, 2]. index 0=z1(no-match/reverse),
      1=z2(match/forward). row i = image i vs 전 텍스트, col j = text j vs 전 이미지.
    direction: 'forward'(z2) | 'reverse'(z1) | 'bidir'(둘).
    temp: 증류 온도 τ (traditional Hinton: softmax(z/τ) + KL ×τ²).
    반환: forward KL(T‖S) 평균 × τ². 채널별(선택)×축(row,col) KL을 평균.
    """
    channels = {"forward": [1], "reverse": [0], "bidir": [0, 1]}.get(direction)
    if channels is None:
        raise ValueError(f"direction must be forward/reverse/bidir, got {direction!r}")

    s = student_logits.float()
    t = teacher_logits.float().detach()

    total = 0.0
    n = 0
    for c in channels:
        ms, mt = s[:, :, c] / temp, t[:, :, c] / temp      # [B, B]
        # i2t: row 분포 (dim=1 = 텍스트 축)
        total = total + F.kl_div(F.log_softmax(ms, dim=1),
                                 F.softmax(mt, dim=1), reduction="batchmean")
        # t2i: col 분포 (transpose 후 dim=1 = 이미지 축)
        total = total + F.kl_div(F.log_softmax(ms.t(), dim=1),
                                 F.softmax(mt.t(), dim=1), reduction="batchmean")
        n += 2
    return (total / n) * (temp ** 2)


def itm_gathered_kd_loss(s_i2t, t_i2t, s_t2i, t_t2i, direction, temp):
    """Phase 2 서브샘플 ITM KD. 각 [B,k+1,2] (0=z1, 1=z2; dim1=k+1 후보, col0=positive).
    gather(i2t,t2i 둘 다)별 × channel(direction)별 dim=1 softmax forward KL(T‖S) 평균 × τ²
    (traditional Hinton: softmax(z/τ) + KL ×τ²; τ~2 중온 regime)."""
    channels = {"forward": [1], "reverse": [0], "bidir": [0, 1]}.get(direction)
    if channels is None:
        raise ValueError(f"direction must be forward/reverse/bidir, got {direction!r}")
    total, n = 0.0, 0
    for s, t in ((s_i2t, t_i2t), (s_t2i, t_t2i)): # 루프를 2번 돈다. 튜플이 길이 2기 때문에
        s = s.float()
        t = t.float().detach()
        for c in channels: # 각 i2t와 t2i는 다르기 때문에 두번 도는 것.
            total = total + F.kl_div(F.log_softmax(s[:, :, c] / temp, dim=1),
                                     F.softmax(t[:, :, c] / temp, dim=1), reduction="batchmean")
            n += 1
    return (total / n) * (temp ** 2)
