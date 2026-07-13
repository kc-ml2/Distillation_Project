import io
import contextlib

from utils import MetricLogger


def test_log_every_prints_elapsed():
    ml = MetricLogger()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in ml.log_every([1, 2, 3], print_freq=1, header="H"):
            pass
    out = buf.getvalue()
    assert "elapsed:" in out
