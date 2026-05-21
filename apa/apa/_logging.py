"""Tiny shared logger for APA pipeline scripts.

Centralizes the timestamped print used throughout the apa.* modules so the
output format is consistent. Each call produces a line of the form:

    [YYYY-MM-DD HH:MM:SS] message
"""
from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    """Print ``message`` prefixed with the current local timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
