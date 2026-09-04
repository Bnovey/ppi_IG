"""Tests for igv.gpu -- CPU-only behaviour, asserted on any host.

The module under test is deliberately usable on a laptop: it must import with
no ``torch`` at all, and its measurement helpers must degrade to ``None``
rather than raise when there is no CUDA device.  Both properties are asserted
here, the first by hiding ``torch`` from the import system and the second via
the ``cpu_only`` fixture.

These tests must NOT infer "no CUDA" from the host: doing so made them pass on
a laptop and fail on the GPU box, which is the only machine where igv.gpu is
load-bearing.  Anything asserting a no-CUDA return value takes ``cpu_only``.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from igv import gpu


# ---------------------------------------------------------------------------
# 1. Import safety and constants
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_min_vram_constant(self):
        assert gpu.MIN_VRAM_GIB == 78.0
        # The alias the rest of the package imports it by is the same value.
        assert gpu.DEFAULT_MIN_VRAM_GIB == 78.0

    def test_no_module_level_torch_import(self):
        """The laptop constraint: importing igv.gpu must not need torch.

        Loads a *fresh, isolated* copy of the module with torch hidden from the
        import machinery.  A module-level ``import torch`` would raise
        ImportError here.  Deliberately not ``importlib.reload(gpu)``: reload
        rebinds new function objects into the live ``igv.gpu``, so any module
        that already did ``from igv.gpu import require_vram`` would keep the
        stale object and identity checks elsewhere in the suite would fail.
        """
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            # Raise before the sys.modules cache is consulted, so an already
            # imported torch is still invisible to the module under test.
            if name == "torch" or name.startswith("torch."):
                raise ImportError("no torch on this box")
            return real_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location(
            "igv_gpu_no_torch", Path(gpu.__file__)
        )
        mod = importlib.util.module_from_spec(spec)
        builtins.__import__ = fake_import
        try:
            spec.loader.exec_module(mod)  # would raise if torch were needed here
            assert mod.MIN_VRAM_GIB == 78.0
            # Every measurement helper stays callable with no torch.
            assert mod.gpu_total_gib() is None
            assert mod.peak_allocated_gib() is None
            assert mod.peak_reserved_gib() is None
            assert mod.current_allocated_gib() is None
            assert mod.reset_peak() is None
            assert mod.has_cuda() is False
            assert mod.oom_error_types() == (RuntimeError,)
            # Without torch a bare RuntimeError is only an OOM if it says so.
            assert mod.is_oom(RuntimeError("CUDA out of memory")) is True
            assert mod.is_oom(RuntimeError("something else")) is False
            with pytest.raises(RuntimeError, match="torch is not installed"):
                mod.require_vram()
            # And the measurement path still yields an honest row.
            with mod.measure_peak("no-torch") as row:
                pass
            assert row["completed"] is True and row["cuda"] is False
        finally:
            builtins.__import__ = real_import
        # The live module is untouched: no reload, nothing rebound.
        assert sys.modules["igv.gpu"] is gpu
        assert "igv_gpu_no_torch" not in sys.modules


# ---------------------------------------------------------------------------
# 2. require_vram
# ---------------------------------------------------------------------------


class TestRequireVram:
    def test_skip_env_returns_nan_without_touching_torch(self, monkeypatch, caplog):
        """IGV_SKIP_VRAM_CHECK=1 must short-circuit before any torch access."""
        monkeypatch.setenv("IGV_SKIP_VRAM_CHECK", "1")

        def explode(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("require_vram touched torch despite the skip")

        monkeypatch.setattr(gpu, "_torch", explode)
        monkeypatch.setattr(gpu, "_cuda_torch", explode)

        with caplog.at_level(logging.WARNING, logger=gpu.log.name):
            out = gpu.require_vram()
        assert math.isnan(out)
        assert "IGV_SKIP_VRAM_CHECK=1" in caplog.text

    def test_skip_return_is_not_a_passing_threshold(self, monkeypatch):
        """NaN, not a number: a caller that compares cannot get a false pass."""
        monkeypatch.setenv("IGV_SKIP_VRAM_CHECK", "1")
        out = gpu.require_vram()
        assert not (out >= gpu.MIN_VRAM_GIB)
        assert not (out < gpu.MIN_VRAM_GIB)

    def test_skip_env_other_values_do_not_skip(self, monkeypatch, cpu_only):
        """Only the exact string "1" is an override."""
        monkeypatch.setenv("IGV_SKIP_VRAM_CHECK", "0")
        with pytest.raises(RuntimeError):
            gpu.require_vram()
        monkeypatch.setenv("IGV_SKIP_VRAM_CHECK", "true")
        with pytest.raises(RuntimeError):
            gpu.require_vram()

    def test_raises_on_cpu_only_box(self, monkeypatch, cpu_only):
        monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
        with pytest.raises(RuntimeError, match="No CUDA device"):
            gpu.require_vram()

    def test_cpu_only_message_names_the_override(self, monkeypatch, cpu_only):
        monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
        with pytest.raises(RuntimeError) as ei:
            gpu.require_vram(min_gib=78.0)
        msg = str(ei.value)
        assert "IGV_SKIP_VRAM_CHECK=1" in msg
        assert "78.0" in msg

    def test_too_small_card_is_rejected(self, monkeypatch):
        """A 40GB A100 must fail the gate; the message must quote its size."""
        monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
        fake = _FakeTorch(total_memory=int(39.5 * 1024 ** 3))
        monkeypatch.setitem(sys.modules, "torch", fake)
        with pytest.raises(RuntimeError, match=r"GPU 0 has 39\.5 GiB"):
            gpu.require_vram()

    def test_large_card_passes_and_returns_gib(self, monkeypatch, caplog):
        """79.2 GiB is the real A100-SXM4-80GB figure -- it must clear 78."""
        monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
        fake = _FakeTorch(total_memory=int(79.2 * 1024 ** 3))
        monkeypatch.setitem(sys.modules, "torch", fake)
        with caplog.at_level(logging.INFO, logger=gpu.log.name):
            gib = gpu.require_vram()
        assert gib == pytest.approx(79.2, abs=0.01)
        assert "GPU 0: 79.2 GiB" in caplog.text

    def test_log_on_success_can_be_silenced(self, monkeypatch, caplog):
        monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
        monkeypatch.setitem(sys.modules, "torch", _FakeTorch(total_memory=100 * 1024 ** 3))
        with caplog.at_level(logging.INFO, logger=gpu.log.name):
            gpu.require_vram(log_on_success=False)
        assert "GPU 0:" not in caplog.text

    def test_total_mem_attribute_error_is_a_clean_runtimeerror(self, monkeypatch):
        """Regression guard for the .total_mem bug in scripts/02_embed_deltas.py.

        ``_CudaDeviceProperties`` has ``total_memory``, not ``total_mem``.
        Whatever goes wrong reading the properties, the caller must see a
        RuntimeError from the gate rather than a bare AttributeError.
        """
        monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
        fake = _FakeTorch(total_memory=None)  # properties object without the attr
        monkeypatch.setitem(sys.modules, "torch", fake)
        with pytest.raises(RuntimeError, match="cannot read GPU 0 properties"):
            gpu.require_vram()

    def test_properties_uses_total_memory_not_total_mem(self, monkeypatch):
        monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
        fake = _FakeTorch(total_memory=80 * 1024 ** 3)
        monkeypatch.setitem(sys.modules, "torch", fake)
        gpu.require_vram()
        assert fake.cuda.props_reads == ["total_memory"]


class _FakeProps:
    """Stand-in for _CudaDeviceProperties that records which attr was read."""

    def __init__(self, total_memory, log):
        self._total_memory = total_memory
        self._log = log

    def __getattr__(self, name):
        self._log.append(name)
        if name == "total_memory" and self.__dict__["_total_memory"] is not None:
            return self.__dict__["_total_memory"]
        raise AttributeError(name)


class _FakeCuda:
    def __init__(self, total_memory, available=True):
        self._total_memory = total_memory
        self._available = available
        self.props_reads: list[str] = []

    def is_available(self):
        return self._available

    def get_device_properties(self, idx):
        return _FakeProps(self._total_memory, self.props_reads)


class _FakeTorch:
    """Minimal torch stand-in: a CUDA device of a chosen size, no real GPU.

    ``available=False`` makes it a CPU-only box regardless of what the host
    actually has -- see the ``cpu_only`` fixture below.
    """

    def __init__(self, total_memory=None, available=True):
        self.cuda = _FakeCuda(total_memory, available=available)


@pytest.fixture
def cpu_only(monkeypatch):
    """Reproduce a CPU-only box on ANY host, including the GPU host.

    igv.gpu reaches torch two ways -- lazily through ``_torch()`` /
    ``_cuda_torch()``, and with a direct ``import torch`` inside
    ``require_vram`` -- and both consult ``sys.modules``, so replacing that one
    entry closes every path.

    Why this exists: the assertions it guards ("No CUDA device", peak counters
    return None, ``has_cuda() is False``) were really assertions about the
    machine running pytest. They passed on a laptop and failed on ``igv-gpu``
    -- i.e. this file was red on the only box where ``igv.gpu`` does any work,
    so the CPU-degradation contract was untested exactly where a regression
    would matter. Forcing the condition tests the code instead of the host.
    """
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(available=False))
    monkeypatch.delenv("IGV_SKIP_VRAM_CHECK", raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# 3. gpu_total_gib and the peak wrappers on CPU
# ---------------------------------------------------------------------------


class TestCpuNoOps:
    def test_all_none_or_noop(self, cpu_only):
        assert gpu.has_cuda() is False
        assert gpu.gpu_total_gib() is None
        assert gpu.reset_peak() is None
        assert gpu.peak_allocated_gib() is None
        assert gpu.peak_reserved_gib() is None
        assert gpu.current_allocated_gib() is None

    def test_gpu_total_gib_rounds_to_2dp(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _FakeTorch(total_memory=85_899_345_920))
        assert gpu.gpu_total_gib() == 80.0


# ---------------------------------------------------------------------------
# 4. is_oom / oom_error_types
# ---------------------------------------------------------------------------


class TestIsOom:
    def test_runtimeerror_by_message(self):
        assert gpu.is_oom(
            RuntimeError("CUDA out of memory. Tried to allocate 1.02 GiB")
        ) is True

    def test_non_oom_errors(self):
        assert gpu.is_oom(ValueError("x")) is False
        assert gpu.is_oom(RuntimeError("shape mismatch")) is False

    def test_torch_oom_type(self):
        torch = pytest.importorskip("torch")
        assert gpu.is_oom(torch.OutOfMemoryError("CUDA out of memory")) is True
        # Type-based, so it holds even when the message says nothing useful.
        assert gpu.is_oom(torch.OutOfMemoryError("")) is True

    def test_oom_error_types_prefers_the_torch_class(self):
        torch = pytest.importorskip("torch")
        types = gpu.oom_error_types()
        assert types == (torch.OutOfMemoryError,)
        # Documented MRO on torch 2.5.1: OutOfMemoryError subclasses RuntimeError.
        assert issubclass(types[0], RuntimeError)


# ---------------------------------------------------------------------------
# 5. measure_peak (contextmanager form)
# ---------------------------------------------------------------------------


class TestMeasurePeak:
    def test_clean_exit(self, cpu_only):
        with gpu.measure_peak("cpu-clean") as row:
            assert row["status"] == "running"
            assert row["completed"] is False
        assert row["label"] == "cpu-clean"
        assert row["status"] == "completed"
        assert row["completed"] is True
        assert row["truncated"] is False
        assert row["oom"] is False
        assert row["error"] is None
        assert isinstance(row["wall_s"], float)
        assert row["cuda"] is False
        # No CUDA -> no numbers, but the row still exists and is honest.
        assert row["peak_allocated_gib"] is None
        assert row["peak_reserved_gib"] is None

    def test_oom_sets_status_and_reraises(self):
        torch = pytest.importorskip("torch")
        row = {}
        with pytest.raises(torch.OutOfMemoryError):
            with gpu.measure_peak("oom") as row:
                raise torch.OutOfMemoryError("CUDA out of memory")
        assert row["status"] == "oom"
        assert row["oom"] is True
        assert row["completed"] is False
        assert row["truncated"] is True
        assert "OutOfMemoryError" in row["error"]

    def test_plain_error_is_error_not_oom(self):
        row = {}
        with pytest.raises(ValueError):
            with gpu.measure_peak("bad") as row:
                raise ValueError("nope")
        assert row["status"] == "error"
        assert row["oom"] is False
        assert row["completed"] is False
        assert row["truncated"] is True
        assert row["error"] == "ValueError: nope"

    def test_error_string_is_bounded(self):
        row = {}
        with pytest.raises(ValueError):
            with gpu.measure_peak() as row:
                raise ValueError("x" * 5000)
        assert len(row["error"]) == 500

    def test_row_is_json_serialisable(self):
        with gpu.measure_peak("json") as row:
            pass
        assert json.loads(json.dumps(row))["status"] == "completed"

    def test_reset_is_called_before_the_body(self, monkeypatch):
        """The reset is the whole point: peak counters are monotonic.

        Order matters -- resetting after the body would report the *previous*
        measurement, which is the silent-no-op failure this module exists to
        prevent.
        """
        calls = []
        monkeypatch.setattr(gpu, "reset_peak", lambda *a, **k: calls.append("reset"))
        with gpu.measure_peak("ordered"):
            calls.append("body")
        assert calls == ["reset", "body"]

    def test_reset_can_be_disabled(self, monkeypatch):
        calls = []
        monkeypatch.setattr(gpu, "reset_peak", lambda *a, **k: calls.append("reset"))
        with gpu.measure_peak("no-reset", reset=False):
            pass
        assert calls == []


# ---------------------------------------------------------------------------
# 6. PeakMemory (class form)
# ---------------------------------------------------------------------------


class TestPeakMemory:
    def test_clean_exit_on_cpu(self, cpu_only):
        with gpu.PeakMemory(label="cpu") as pm:
            assert pm is not None
        assert pm.peak_gib is None
        assert pm.reserved_gib is None
        assert pm.completed is True
        assert pm.oom is False
        assert pm.truncated is False
        assert pm.status == "completed"
        assert pm.error is None
        assert pm.label == "cpu"
        assert isinstance(pm.wall_s, float)

    def test_positional_signature_matches_the_contract(self):
        """device, reset, suppress_oom, label -- in that order."""
        pm = gpu.PeakMemory(0, True, False, "positional")
        assert (pm.device, pm.reset, pm.suppress_oom, pm.label) == (
            0, True, False, "positional",
        )

    def test_normal_exception_propagates_and_is_recorded(self):
        pm = gpu.PeakMemory(label="boom")
        with pytest.raises(ValueError, match="nope"):
            with pm:
                raise ValueError("nope")
        assert pm.completed is False
        assert pm.oom is False
        assert pm.truncated is True
        assert pm.status == "error"
        assert pm.error == repr(ValueError("nope"))

    def test_oom_suppressed_when_asked(self):
        """A sweep must be able to record the OOM and continue."""
        with gpu.PeakMemory(label="oom", suppress_oom=True) as pm:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.02 GiB")
        assert pm.oom is True
        assert pm.completed is False
        assert pm.truncated is True
        assert pm.status == "oom"
        assert "out of memory" in pm.error

    def test_oom_reraised_by_default(self):
        pm = gpu.PeakMemory(label="oom")
        with pytest.raises(RuntimeError, match="out of memory"):
            with pm:
                raise RuntimeError("CUDA out of memory")
        assert pm.oom is True
        assert pm.completed is False

    def test_suppress_oom_does_not_swallow_real_bugs(self):
        """suppress_oom is not a bare except: a ValueError must still surface."""
        pm = gpu.PeakMemory(label="bug", suppress_oom=True)
        with pytest.raises(ValueError):
            with pm:
                raise ValueError("a real bug")
        assert pm.oom is False
        assert pm.status == "error"

    def test_torch_oom_type_is_suppressed_too(self):
        torch = pytest.importorskip("torch")
        with gpu.PeakMemory(label="t", suppress_oom=True) as pm:
            raise torch.OutOfMemoryError("CUDA out of memory")
        assert pm.oom is True
        assert pm.status == "oom"

    def test_as_dict_json_serialises(self, cpu_only):
        with gpu.PeakMemory(label="row") as pm:
            pass
        blob = json.dumps(pm.as_dict())
        back = json.loads(blob)
        assert back["label"] == "row"
        assert back["completed"] is True
        assert back["truncated"] is False
        assert back["peak_gib"] is None
        assert back["oom"] is False

    def test_as_dict_json_serialises_after_failure(self):
        pm = gpu.PeakMemory(label="row", suppress_oom=True)
        with pm:
            raise RuntimeError("CUDA out of memory")
        back = json.loads(json.dumps(pm.as_dict()))
        assert back["status"] == "oom"
        assert back["truncated"] is True
        assert isinstance(back["error"], str)

    def test_reset_calls_reset_peak_with_the_device(self, monkeypatch, cpu_only):
        seen = []
        monkeypatch.setattr(gpu, "reset_peak", lambda dev=None: seen.append(dev))
        with gpu.PeakMemory(device=3, reset=True):
            pass
        assert seen == [3]

    def test_reset_false_skips_the_reset(self, monkeypatch):
        seen = []
        monkeypatch.setattr(gpu, "reset_peak", lambda dev=None: seen.append(dev))
        with gpu.PeakMemory(device=0, reset=False):
            pass
        assert seen == []

    def test_repr_is_readable(self):
        with gpu.PeakMemory(label="r") as pm:
            pass
        assert "PeakMemory(label='r'" in repr(pm)
        assert "status='completed'" in repr(pm)
