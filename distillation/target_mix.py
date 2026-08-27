import torch.nn.functional as F
def ttm_gamma(global_step, hold_steps, decay_end_steps):
    """teacher↔momentum soft-slot 분배 γ ∈ [0,1].
    γ=1 (홀드), hold~decay_end 선형 1→0, 이후 0. decay_end>hold 전제."""
    if global_step < hold_steps:
        return 1.0
    if global_step >= decay_end_steps:
        return 0.0
    return 1.0 - (global_step - hold_steps) / (decay_end_steps - hold_steps)


def teacher_soft_queue(row_feat, col_all, tau):
    """variant C: 학생 후보군과 정렬된 col_all[D, B+queue] 위 티처 softmax [B, B+queue]."""
    sim = (row_feat @ col_all) / tau                  # [B, B+queue]
    return F.softmax(sim, dim=1)


def mix_target(onehot, momentum_soft, teacher_soft, gamma, soft_weight):
    """(1-W)·onehot + W·(γ·teacher + (1-γ)·momentum). 모든 인자 [B,N] 행분포."""
    soft = gamma * teacher_soft + (1.0 - gamma) * momentum_soft
    return (1.0 - soft_weight) * onehot + soft_weight * soft


def enqueue_all(pairs, ptr, bs, queue_size):
    """각 (queue[D,Q], feats_T[D,bs])를 동일 ptr의 열에 써서 큐 간 정렬 보장.
    ptr은 한 번만 전진. momentum·teacher 큐를 이 함수로 함께 넣어야 열이 일치한다."""
    for queue, feats_T in pairs:
        queue[:, ptr:ptr + bs] = feats_T
    return (ptr + bs) % queue_size
