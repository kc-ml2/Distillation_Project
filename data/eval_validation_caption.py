import os


def parse_cpu_list(spec):
    """'0-31' 또는 '0-3,8,10-12' -> 정렬 불필요한 코어 집합. None/'' -> None."""
    if not spec:
        return None
    cores = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            cores.update(range(int(a), int(b) + 1))
        else:
            cores.add(int(part))
    return cores or None


class cpu_affinity:
    """채점 구간 동안 프로세스(및 SPICE의 java 자식)를 지정 코어에 제한하고
    종료 시 원복. cores=None 또는 플랫폼 미지원이면 no-op."""

    def __init__(self, cores):
        self.cores = cores
        self.orig = None

    def __enter__(self):
        if self.cores and hasattr(os, "sched_setaffinity"):
            self.orig = os.sched_getaffinity(0)
            os.sched_setaffinity(0, self.cores)
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            os.sched_setaffinity(0, self.orig)
        return False
