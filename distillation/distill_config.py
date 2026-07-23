"""Teacher keep-set derivation and itm_target_mix config validation."""


def derive_teacher_keep(distill_cfg):
    """Which teacher submodule paths must be kept, given the distill config.
    itc/lm: their own enabled flags. itc_target_mix: 'itc' whenever enabled (teacher
    ITC feats feed the γ-mixed target). itm_target_mix: 'itc' when neg_source=='teacher'
    (selection feats), 'itm' when soft_weight>0 (teacher scoring)."""
    keep = set()
    if distill_cfg.get('itc', {}).get('enabled', False):
        keep.add('itc')
    if distill_cfg.get('lm', {}).get('enabled', False):
        keep.add('lm')
    if distill_cfg.get('itc_target_mix', {}).get('enabled', False):
        keep.add('itc')
    itm = distill_cfg.get('itm_target_mix', {})
    if itm.get('enabled', False):
        if itm.get('neg_source') == 'teacher':
            keep.add('itc')
        if float(itm.get('soft_weight', 0.0)) > 0.0:
            keep.add('itm')
    return tuple(sorted(keep))


def validate_itm_mix_config(distill_cfg):
    """Assert itm_target_mix config is well-formed (no-op when disabled/absent)."""
    itm = distill_cfg.get('itm_target_mix', {})
    if not itm.get('enabled', False):
        return
    ns = itm.get('neg_source')
    assert ns in ('teacher', 'student'), \
        f"itm_target_mix.neg_source must be 'teacher'|'student', got {ns!r}"
    w = float(itm.get('soft_weight', 0.0))
    assert 0.0 <= w <= 1.0, f"itm_target_mix.soft_weight must be in [0,1], got {w}"
    sched = itm.get('schedule', 'constant')
    assert sched == 'constant', \
        f"itm_target_mix.schedule only 'constant' implemented, got {sched!r}"
