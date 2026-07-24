import torch
import torch.nn.functional as F


def itc_distill_loss(image_feat_s, text_feat_s, teacher_img_feat, teacher_txt_feat, temp):
    """In-batch B×B relational KD for the ITC signal.

    Args:
        image_feat_s, text_feat_s: student online features, [B, D], L2-normalized, require grad.
        teacher_img_feat, teacher_txt_feat: teacher features, [B, D], L2-normalized (constants).
        temp: distillation temperature τ (same for teacher and student).

    Returns:
        Scalar tensor = 0.5 * (KL_i2t + KL_t2i) * τ, with KL(teacher ‖ student).

    Note: KD grad ∂L/∂z = (q−p)/τ. 여기 τ≪1(sharpening) saturation regime에선 (q−p)가 유계라
    grad ∝ 1/τ. Hinton의 × τ² 보정은 고온(선형화) 전제에서만 grad를 O(1)로 되돌리며,
    이 regime엔 과보정이라 grad를 τ배로 죽인다(2026-07-07 keeptau2 = no-op으로 실증).
    → 이 regime의 올바른 정규화는 **× τ** (grad ∝ τ·(1/τ) = O(1)). 이러면 grad 크기가 τ에 무관해져
    두 arm이 단일 λ(weight)로 자동 정합된다. crossover 실측·유도: critical_bugfix/2026-07-14.
    크기 정합(ITC 대비 목표 ratio)은 λ_itc(weight)로 잡는다.
    """
    teacher_img_feat = teacher_img_feat.detach()
    teacher_txt_feat = teacher_txt_feat.detach()

    s_s = (image_feat_s @ text_feat_s.t()) / temp          # [B, B] student
    s_t = (teacher_img_feat @ teacher_txt_feat.t()) / temp  # [B, B] teacher

    # image->text: rows = images, distribution over texts (dim=1)
    loss_i2t = F.kl_div(F.log_softmax(s_s, dim=1), F.softmax(s_t, dim=1), reduction="batchmean")
    # text->image: transpose
    loss_t2i = F.kl_div(F.log_softmax(s_s.t(), dim=1), F.softmax(s_t.t(), dim=1), reduction="batchmean")

    return 0.5 * (loss_i2t + loss_t2i) * temp


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


def itm_target_mix_loss(vl_output, itm_labels, teacher_soft, soft_weight):
    """Soft-target CE for ITM: target = (1-W)*onehot(itm_labels) + W*teacher_soft.

    vl_output   : [N, 2] student ITM logits.
    itm_labels  : [N]    long class indices in {0=no-match, 1=match}.
    teacher_soft: [N, 2] teacher match distribution (rows sum to 1, no grad).
    soft_weight : W in [0, 1]. W=0 reduces exactly to F.cross_entropy(vl_output, itm_labels).
    """
    onehot = F.one_hot(itm_labels, num_classes=2).to(vl_output.dtype)
    target = (1.0 - soft_weight) * onehot + soft_weight * teacher_soft.to(vl_output.dtype)
    return -(target * F.log_softmax(vl_output, dim=1)).sum(dim=1).mean()


def itm_hinton_kd_loss(vl_output, itm_labels, teacher_soft, alpha, temp):
    """2-term Hinton KD for ITM (스킴 B): 하드 CE 앵커 + tempered 티처 KL.

        loss = (1-alpha)*CE(z, itm_labels) + alpha * T^2 * KL(teacher_soft || softmax(z/T))

    vl_output   : [N, 2] student ITM logits (z).
    itm_labels  : [N]    long {0=no-match, 1=match}.
    teacher_soft: [N, 2] = softmax(teacher_logits / temp), no grad. ×T^2 스케일이 성립하려면
                  반드시 '동일 temp'로 tempered여야 한다 (OnlineTeacher.itm_soft(temp=T)).
    alpha       : soft 항 가중(α). α=0이면 순수 하드 CE.
    temp        : T. KL 항에서 student도 /T로 tempered.

    itm_target_mix(확률 믹싱)와 달리 하드 라벨을 target에 '녹이지' 않고 CE를 별도 항으로
    유지 → 갭 sharpening을 안 죽인다(균형 gap→teacher gap). lm_distill_loss의 KL·×T² 관례를 ITM에 적용.
    """
    hard = F.cross_entropy(vl_output, itm_labels)
    kd = F.kl_div(F.log_softmax(vl_output / temp, dim=1),
                  teacher_soft.to(vl_output.dtype),
                  reduction='batchmean') * (temp ** 2)
    return (1.0 - alpha) * hard + alpha * kd
