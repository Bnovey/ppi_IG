"""Provenance sidecars, so every artifact says how it was made.

Every stage script writes its output alongside a ``<output>.prov.json`` sidecar
recording inputs, parameters and environment. Analysis code then calls
:func:`assert_provenance` to check that an artifact was produced by the arm it
is about to be interpreted as.

This exists because the predecessor project's two worst bugs were silent
no-ops: a default argument routed every attribution to one model while the
docs claimed a three-model ensemble, and a misordered threshold made a filter
inert for an entire 35-hour campaign. Neither raised an error; both were found
by auditing artifacts afterwards. Checking the recorded arm instead of the
requested one is the cheapest defence available.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".prov.json"


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _git_dirty(repo: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def _package_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def environment() -> dict[str, Any]:
    """Capture the bits of the environment that have historically broken runs.

    ``torch`` and ``boltz`` versions are recorded because installing certain
    packages silently replaces the pinned CUDA torch wheel, and the resulting
    failures never mention torch.
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": _package_version("torch"),
        "boltz": _package_version("boltz"),
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_gib"] = round(props.total_memory / 1024**3, 1)
    except Exception:
        env["cuda_available"] = None
    return env


@dataclass
class Provenance:
    """What produced an artifact.

    ``arm`` is the load-bearing field: it names the experimental arm, e.g.
    ``{"score": "iptm", "trunk": "full", "method": "ig", "m_steps": 15}``.
    Analysis asserts on it rather than trusting the flags it passed in.
    """

    stage: str
    output: str
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    arm: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    env: dict[str, Any] = field(default_factory=environment)
    git_commit: str | None = None
    git_dirty: bool | None = None
    argv: list[str] = field(default_factory=lambda: list(sys.argv))

    def __post_init__(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        if self.git_commit is None:
            self.git_commit = _git_commit(repo)
        if self.git_dirty is None:
            self.git_dirty = _git_dirty(repo)


def sidecar_path(output: str | os.PathLike[str]) -> Path:
    return Path(str(output) + SIDECAR_SUFFIX)


def write(
    output: str | os.PathLike[str],
    stage: str,
    *,
    inputs: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    arm: dict[str, Any] | None = None,
    notes: str | None = None,
) -> Path:
    """Write a sidecar next to ``output``. Returns the sidecar path."""
    prov = Provenance(
        stage=stage,
        output=str(output),
        inputs=inputs or {},
        params=params or {},
        arm=arm or {},
        notes=notes,
    )
    path = sidecar_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: a truncated sidecar is worse than none, because it
    # reads as "provenance exists" while carrying nothing.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(asdict(prov), fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read(output: str | os.PathLike[str]) -> dict[str, Any]:
    path = sidecar_path(output)
    if not path.exists():
        raise FileNotFoundError(
            f"No provenance sidecar for {output}. Artifacts without provenance "
            "are not interpretable; regenerate it with the stage script."
        )
    with open(path) as fh:
        return json.load(fh)


def assert_provenance(output: str | os.PathLike[str], **expected: Any) -> dict[str, Any]:
    """Check an artifact's recorded ``arm`` against what the caller expects.

    Raises ``AssertionError`` naming every mismatch. Keys absent from the
    recorded arm are themselves a mismatch -- an unrecorded arm cannot be
    verified, so it is treated as wrong rather than assumed right.
    """
    prov = read(output)
    arm = prov.get("arm", {})
    problems = []
    for key, want in expected.items():
        if key not in arm:
            problems.append(f"{key}: not recorded (expected {want!r})")
        elif arm[key] != want:
            problems.append(f"{key}: recorded {arm[key]!r}, expected {want!r}")
    if problems:
        raise AssertionError(
            f"Provenance mismatch for {output}:\n  " + "\n  ".join(problems)
        )
    return prov
