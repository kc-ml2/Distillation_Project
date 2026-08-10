"""Teacher keep-set derivation for the distillation online teacher."""


def derive_teacher_keep(distill_cfg):
    """Which teacher submodule paths must be kept, given the distill config.
    lm: its own enabled flag. itc_target_mix: 'itc' whenever enabled (teacher ITC
    feats feed the γ-mixed target). itm (k=4 traditional KD): 'itm' whenever enabled
    (teacher itm_matrix scoring; negatives are picked from student sim so no 'itc')."""
    keep = set()
    if distill_cfg.get('lm', {}).get('enabled', False):
        keep.add('lm')
    if distill_cfg.get('itc_target_mix', {}).get('enabled', False):
        keep.add('itc')
    if distill_cfg.get('itm', {}).get('enabled', False):
        keep.add('itm')
    return tuple(sorted(keep))


def need_teacher_image_embeds(need_teacher_itc, lm_kd_enabled, itm_kd_enabled):
    """Whether this training step needs the teacher's image_embeds for ANY of
    itc_feats/lm_logits/itm_matrix — decides whether OnlineTeacher.encode_image()
    should be called once this step (see pretrain.py train loop)."""
    return bool(need_teacher_itc or lm_kd_enabled or itm_kd_enabled)
