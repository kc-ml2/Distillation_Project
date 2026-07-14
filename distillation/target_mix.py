import torch
import torch.nn.functional as F


def ttm_gamma(global_step, hold_steps, decay_end_steps):
    """teacher↔momentum soft-slot 분배 γ ∈ [0,1].
    γ=1 (홀드), hold~decay_end 선형 1→0, 이후 0. decay_end>hold 전제."""
    if global_step < hold_steps:
        return 1.0
    if global_step >= decay_end_steps:
        return 0.0
    return 1.0 - (global_step - hold_steps) / (decay_end_steps - hold_steps)
