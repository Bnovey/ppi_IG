"""Tests for igv.boltz_score checkpoint selection.

These are deliberately torch-free: select_checkpoint is the one part of
boltz_score that can run on CPU without the GPU container.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.boltz_score import select_checkpoint  # noqa: E402


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_bytes(b"")


def test_prefers_confidence_over_affinity(tmp_path):
    """The regression this module exists for.

    A full download_boltz2 leaves both checkpoints in the cache, and
    boltz2_aff.ckpt sorts first. Selecting it would attribute gradients of the
    affinity head while every score here reads the confidence head -- silently,
    with no error.
    """
    _touch(tmp_path, "boltz2_aff.ckpt", "boltz2_conf.ckpt")
    assert select_checkpoint(tmp_path).name == "boltz2_conf.ckpt"
    # Guard the exact failure mode: never the alphabetically-first file.
    assert select_checkpoint(tmp_path).name != sorted(
        p.name for p in tmp_path.glob("*.ckpt")
    )[0]


def test_single_confidence_checkpoint(tmp_path):
    _touch(tmp_path, "boltz2_conf.ckpt")
    assert select_checkpoint(tmp_path).name == "boltz2_conf.ckpt"


def test_affinity_only_raises_rather_than_loading_wrong_model(tmp_path):
    _touch(tmp_path, "boltz2_aff.ckpt")
    with pytest.raises(FileNotFoundError, match="confidence checkpoint"):
        select_checkpoint(tmp_path)


def test_no_checkpoints_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .ckpt files"):
        select_checkpoint(tmp_path)


def test_unrecognised_single_checkpoint_is_accepted(tmp_path):
    """A custom/fine-tuned checkpoint should still load."""
    _touch(tmp_path, "my_finetuned.ckpt")
    assert select_checkpoint(tmp_path).name == "my_finetuned.ckpt"
