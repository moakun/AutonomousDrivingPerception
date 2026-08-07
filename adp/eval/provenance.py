"""Stamp results files with what produced them (improvement #6).

Without this, two JSONs in out/eval/ are indistinguishable six months later:
same shape, different code, no way to tell which. Every harness now records
git SHA, dirty state, timestamp, host, and the config fingerprint, so REPORT.md
is reproducible in fact rather than in principle.
"""

from __future__ import annotations

import datetime
import platform
import subprocess

from adp.config import CONFIG


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance() -> dict:
    sha = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_dirty": bool(status) if status is not None else None,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "config_fingerprint": CONFIG.fingerprint(),
    }


def stamp(report: dict) -> dict:
    """Attach provenance in place and return the report (chainable)."""
    report["_provenance"] = provenance()
    return report
