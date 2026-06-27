import torch
import torch.nn.functional as F


def itc_distill_loss(image_feat_s, text_feat_s, teacher_img_feat, teacher_txt_feat, temp):
    """In-batch B×B relational KD for the ITC signal.

    Args:
        image_feat_s, text_feat_s: student online features, [B, D], L2-normalized, require grad.
        teacher_img_feat, teacher_txt_feat: teacher features, [B, D], L2-normalized (constants).
        temp: distillation temperature τ (same for teacher and student).

    Returns:
        Scalar tensor = 0.5 * (KL_i2t + KL_t2i) * τ², with KL(teacher ‖ student).
    """
    teacher_img_feat = teacher_img_feat.detach()
    teacher_txt_feat = teacher_txt_feat.detach()

    s_s = (image_feat_s @ text_feat_s.t()) / temp          # [B, B] student
    s_t = (teacher_img_feat @ teacher_txt_feat.t()) / temp  # [B, B] teacher

    # image->text: rows = images, distribution over texts (dim=1)
    loss_i2t = F.kl_div(F.log_softmax(s_s, dim=1), F.softmax(s_t, dim=1), reduction="batchmean")
    # text->image: transpose
    loss_t2i = F.kl_div(F.log_softmax(s_s.t(), dim=1), F.softmax(s_t.t(), dim=1), reduction="batchmean")

    return 0.5 * (loss_i2t + loss_t2i) * (temp ** 2)
