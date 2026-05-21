"""Smoke tests for apa._logging.

Asserts the timestamped log helper produces the format the inline _log
helpers it replaces produced (``[YYYY-MM-DD HH:MM:SS] message`` + newline,
flushed). This guards against accidental format drift during refactors.
"""
from __future__ import annotations

import contextlib
import io
import re

from apa._logging import log


_TS_RE = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]"


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def test_log_format_basic():
    out = _capture(log, "hello")
    assert re.fullmatch(rf"{_TS_RE} hello\n", out), repr(out)


def test_log_format_with_special_chars():
    out = _capture(log, "K=8 lr=0.5 -> 90.05% acc")
    assert re.fullmatch(rf"{_TS_RE} K=8 lr=0\.5 -> 90\.05% acc\n", out), repr(out)


def test_log_format_empty():
    out = _capture(log, "")
    assert re.fullmatch(rf"{_TS_RE} \n", out), repr(out)
