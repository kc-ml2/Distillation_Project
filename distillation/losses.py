import torch
import torch.nn.functional as F


def itc_distill_loss(image_feat_s, text_feat_s, teacher_img_feat, teacher_txt_feat, temp):
    """In-batch B×B relational KD for the ITC signal.

    Args:
        image_feat_s, text_feat_s: student online features, [B, D], L2-normalized, require grad.
        teacher_img_feat, teacher_txt_feat: teacher features, [B, D], L2-normalized (constants).
        temp: distillation temperature τ (same for teacher and student).

    Returns:
        Scalar tensor = 0.5 * (KL_i2t + KL_t2i), with KL(teacher ‖ student).

    Note: Hinton의 × T² 보정은 T>1(softening) 전제에서만 gradient를 order-1로 되돌린다.
    ITC-KD의 τ는 CLIP식 contrastive 온도로 τ≪1(sharpening)이라, × τ²는 보정이 아니라
    gradient를 τ배로 죽이는 역효과(2026-07-07 측정·분석: critical_bugfix/). → 제거.
    크기 정합은 loss 합산 시 λ_itc(weight)로 잡는다.
    """
    teacher_img_feat = teacher_img_feat.detach()
    teacher_txt_feat = teacher_txt_feat.detach()

    s_s = (image_feat_s @ text_feat_s.t()) / temp          # [B, B] student
    s_t = (teacher_img_feat @ teacher_txt_feat.t()) / temp  # [B, B] teacher

    # image->text: rows = images, distribution over texts (dim=1)
    loss_i2t = F.kl_div(F.log_softmax(s_s, dim=1), F.softmax(s_t, dim=1), reduction="batchmean")
    # text->image: transpose
    loss_t2i = F.kl_div(F.log_softmax(s_s.t(), dim=1), F.softmax(s_t.t(), dim=1), reduction="batchmean")

    return 0.5 * (loss_i2t + loss_t2i)


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
    assert student_logits.shape[-1] == teacher_logits.shape[-1], (
        f"vocab size mismatch: student {student_logits.shape[-1]} "
        f"vs teacher {teacher_logits.shape[-1]}"
    )
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :].detach()
    valid = decoder_targets[:, 1:] != -100      # [B, L-1], CE가 학습하는 위치와 동일 집합
    s = s[valid]                                 # [N_valid, V]
    t = t[valid]
    return F.kl_div(
        F.log_softmax(s / temp, dim=-1),
        F.softmax(t / temp, dim=-1),
        reduction="batchmean",                   # N_valid로 나눔 = 유효 토큰당 평균
    ) * (temp ** 2)
