"""GPU capability gate and peak-memory instrumentation.

Two jobs, both of which used to be missing or duplicated:

1. :func:`require_vram` -- one copy of the VRAM gate that four GPU stages
   (``scripts/02_embed_deltas.py``, ``03_attribute.py``, ``04_scan.py``,
   ``07_sanity.py``) each carried privately.  One of those copies read
   ``.total_mem``, which does not exist on ``_CudaDeviceProperties``, so it
   raised ``AttributeError`` instead of gating.  The attribute is
   ``total_memory``.

2. :func:`measure_peak` / :class:`PeakMemory` -- peak-VRAM measurement that
   records *whether the run finished*.  Nothing in the tracked repo measured
   peak memory at all: ``igv.attrib.free_cuda_memory`` only gc/syncs/empties
   the cache and never touches the peak counters, and
   ``torch.cuda.max_memory_allocated()`` is process-global and monotonic until
   an explicit ``reset_peak_memory_stats()``.  A sweep that omits the reset
   reports the same largest-so-far number on every row and looks entirely
   plausible -- so the reset happens here, once, where it cannot be forgotten.

**Import-safe with no torch and no CUDA.**  There is no module-level
``import torch``; every function imports it lazily.  On a CPU-only box the
measurement helpers return ``None`` or are no-ops rather than raising, so call
sites need no ``if torch.cuda.is_available()`` guards.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any, Iterator

log = logging.getLogger(__name__)

# 78, not 80: an 80GB-class card reports 81920 MiB to nvidia-smi but
# torch's total_memory returns the usable framebuffer after the ECC/reserve
# carve-out -- 79.2 GiB on A100-SXM4-80GB. A gate of 80 is unreachable on
# the exact hardware this project targets. 78 still rejects a 40GB A100.
# Mirrors scripts/cloud/bootstrap.sh:19 MIN_VRAM_GIB=78 -- keep the two in step.
MIN_VRAM_GIB: float = 78.0

# Alias under the name the rest of the package imports it by.  Same object;
# both spellings are supported so no call site has to be rewritten.
DEFAULT_MIN_VRAM_GIB: float = MIN_VRAM_GIB

#: Env var that turns the gate into a warning.  Pre-existing; honoured by all
#: former per-script copies.
SKIP_ENV = "IGV_SKIP_VRAM_CHECK"

_GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# lazy torch access
# ---------------------------------------------------------------------------
def _torch() -> Any | None:
    """Return the ``torch`` module, or ``None`` if it is not installed."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_torch() -> Any | None:
    """Return ``torch`` only if a CUDA device is actually usable, else ``None``."""
    torch = _torch()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
    except (AttributeError, RuntimeError):
        return None
    return torch


def has_cuda() -> bool:
    """True iff torch is importable and reports a usable CUDA device."""
    return _cuda_torch() is not None


# ---------------------------------------------------------------------------
# capability gate
# ---------------------------------------------------------------------------
def gpu_total_gib() -> float | None:
    """Total VRAM of GPU 0 in GiB (2dp), or ``None`` if there is no CUDA device.

    The attribute is ``total_memory``.  ``total_mem`` does **not** exist on
    ``_CudaDeviceProperties`` -- reading it raises ``AttributeError``, which is
    exactly the live bug this function replaces.
    """
    torch = _cuda_torch()
    if torch is None:
        return None
    try:
        return round(torch.cuda.get_device_properties(0).total_memory / _GIB, 2)
    except (AttributeError, RuntimeError):
        return None


def require_vram(
    min_gib: float = DEFAULT_MIN_VRAM_GIB,
    *,
    log_on_success: bool = True,
) -> float:
    """Fail fast rather than OOM 30 minutes in.  Returns the measured GiB.

    ``IGV_SKIP_VRAM_CHECK=1`` logs a warning and returns ``float("nan")``
    without touching torch, so the stage runs unguarded at the caller's risk.
    A NaN return is deliberately *not* a number you can compare a threshold
    against -- ``nan < 78`` is False and ``nan >= 78`` is False -- so a caller
    that forgets the override cannot silently treat it as "enough VRAM".
    Use ``math.isnan(...)`` to detect the skip.

    Behaviour change from the four per-script copies this replaces: the skip is
    logged at WARNING everywhere.  ``scripts/03_attribute.py`` used to skip
    silently and ``scripts/02_embed_deltas.py`` logged it at INFO; both now
    warn.  The success line (``"GPU 0: %.1f GiB"``) preserves
    ``scripts/04_scan.py:84``, previously the only place the measured VRAM was
    printed.

    Raises:
        RuntimeError: torch missing, no CUDA device, or less than ``min_gib``.
    """
    if os.environ.get(SKIP_ENV) == "1":
        log.warning("%s=1 -- skipping the %s GiB check", SKIP_ENV, min_gib)
        return float("nan")

    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is not installed; cannot check VRAM") from None

    try:
        available = bool(torch.cuda.is_available())
    except AttributeError:
        # A torch build without the cuda submodule/attribute is as good as no GPU.
        raise RuntimeError("torch is not installed; cannot check VRAM") from None

    if not available:
        raise RuntimeError(
            f"No CUDA device. This stage needs >= {min_gib} GiB VRAM. "
            f"Set {SKIP_ENV}=1 to override."
        )

    try:
        gib = round(torch.cuda.get_device_properties(0).total_memory / _GIB, 2)
    except AttributeError as exc:  # e.g. the .total_mem class of bug
        raise RuntimeError(f"cannot read GPU 0 properties: {exc!r}") from None

    if gib < min_gib:
        raise RuntimeError(
            f"GPU 0 has {gib:.1f} GiB; this project needs >= {min_gib} GiB. "
            "Full-trunk Boltz-2 work fills an entire 80 GB card. "
            f"Set {SKIP_ENV}=1 to override at your own risk."
        )
    if log_on_success:
        log.info("GPU 0: %.1f GiB", gib)
    return gib


# ---------------------------------------------------------------------------
# thin peak-counter wrappers -- no-ops / None without CUDA
# ---------------------------------------------------------------------------
def reset_peak(device: Any = None) -> None:
    """Zero the process-global peak counters.  No-op without CUDA.

    Must be called before every measurement: ``max_memory_allocated`` is
    monotonic until it is reset, so a sweep without this reports the
    largest-so-far figure for every point.
    """
    torch = _cuda_torch()
    if torch is None:
        return
    with contextlib.suppress(RuntimeError):
        torch.cuda.reset_peak_memory_stats(device)


def peak_allocated_gib(device: Any = None) -> float | None:
    """Peak *PyTorch-allocated* bytes since the last reset, in GiB.

    Excludes the CUDA context and any non-PyTorch allocation -- roughly 0.8 GiB
    on this project's runs (the recorded OOM in
    ``results/sanity_4fqi_h1_complex_pde.json`` shows 79.21 GiB in use against
    78.43 GiB allocated by PyTorch).  Report it alongside
    :func:`peak_reserved_gib`; neither alone is the answer.
    """
    torch = _cuda_torch()
    if torch is None:
        return None
    return torch.cuda.max_memory_allocated(device) / _GIB


def peak_reserved_gib(device: Any = None) -> float | None:
    """Peak bytes reserved by the caching allocator, in GiB.

    With ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
    (``docker/Dockerfile:6``) this is a poor proxy for what the allocator can
    satisfy: expandable segments grow and are reused rather than being carved
    into fixed blocks, so reserved no longer bounds the largest servable
    request.  Recorded, not trusted on its own.
    """
    torch = _cuda_torch()
    if torch is None:
        return None
    return torch.cuda.max_memory_reserved(device) / _GIB


def current_allocated_gib(device: Any = None) -> float | None:
    """Bytes currently allocated by PyTorch, in GiB.  ``None`` without CUDA."""
    torch = _cuda_torch()
    if torch is None:
        return None
    return torch.cuda.memory_allocated(device) / _GIB


def _synchronize() -> None:
    """Best-effort ``cuda.synchronize``; tolerates a GPU left in a bad state.

    Same tolerance as ``igv.attrib.free_cuda_memory``: after an OOM the device
    may be unusable, and cleanup must never mask the original error.
    """
    torch = _cuda_torch()
    if torch is None:
        return
    with contextlib.suppress(RuntimeError):
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# OOM classification
# ---------------------------------------------------------------------------
def oom_error_types() -> tuple[type, ...]:
    """Exception types that mean "the allocator hit the wall".

    On torch 2.5.1 both ``torch.OutOfMemoryError`` and
    ``torch.cuda.OutOfMemoryError`` exist and are the same class, with MRO
    ``(OutOfMemoryError, RuntimeError, Exception, BaseException, object)``.
    Older builds raise a plain ``RuntimeError``; :func:`is_oom` covers those by
    message.  Without torch the only thing we can name is ``RuntimeError``.
    """
    torch = _torch()
    if torch is None:
        return (RuntimeError,)
    exc = getattr(torch, "OutOfMemoryError", None) or getattr(
        torch.cuda, "OutOfMemoryError", None
    )
    return (exc,) if exc is not None else (RuntimeError,)


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure, by type or by message."""
    if isinstance(exc, oom_error_types()):
        # A bare RuntimeError is only an OOM if it says so (see below).
        if type(exc) is RuntimeError:
            return "out of memory" in str(exc).lower()
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
_TRUNCATED_DOC = """
``truncated=True`` means the body did not finish.  A peak recorded that way is
the allocator hitting the wall on a run that died, i.e. a **lower bound**, and
must never be reported as a memory requirement: a run that gets further may
reveal *more* requirement, not less.  Only ``truncated=False`` rows are
evidence of what the stage actually costs.
"""


@contextlib.contextmanager
def measure_peak(label: str = "", *, reset: bool = True) -> Iterator[dict]:
    """Record peak VRAM for a block, plus an honest completed-vs-OOM status.

    Yields a mutable dict (the pattern ``offload_large_saved_tensors`` already
    uses) that is filled in on the way out, so the caller keeps the row even
    when the body raises::

        row = None
        try:
            with measure_peak(f"L={L}") as row:
                run_backward()
        except Exception:
            pass          # row["status"] is "oom" or "error"

    Keys: ``label status completed oom truncated peak_allocated_gib
    peak_reserved_gib wall_s error cuda``.

    Exceptions are re-raised after the row is filled -- this measures, it does
    not swallow.  Use :class:`PeakMemory` with ``suppress_oom=True`` for a
    sweep that should keep going past an OOM.
    """
    info: dict = {
        "label": label,
        "status": "running",
        "completed": False,
        "oom": False,
        "truncated": True,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
        "wall_s": None,
        "error": None,
        "cuda": has_cuda(),
    }
    if reset:
        # Synchronize first: pending kernels would otherwise land their
        # allocations after the reset and be counted against this block.
        _synchronize()
        reset_peak()
    t0 = time.perf_counter()
    try:
        yield info
    except BaseException as exc:  # noqa: BLE001 -- re-raised below
        oom = is_oom(exc)
        info["status"] = "oom" if oom else "error"
        info["oom"] = oom
        info["completed"] = False
        info["truncated"] = True
        info["error"] = f"{type(exc).__name__}: {exc}"[:500]
        raise
    else:
        info["status"] = "completed"
        info["completed"] = True
        info["truncated"] = False
    finally:
        _synchronize()
        info["peak_allocated_gib"] = peak_allocated_gib()
        info["peak_reserved_gib"] = peak_reserved_gib()
        info["wall_s"] = time.perf_counter() - t0
        log.info(
            "peak[%s] status=%s allocated=%s GiB reserved=%s GiB wall=%.1f s",
            info["label"],
            info["status"],
            _fmt(info["peak_allocated_gib"]),
            _fmt(info["peak_reserved_gib"]),
            info["wall_s"],
        )


measure_peak.__doc__ = (measure_peak.__doc__ or "") + _TRUNCATED_DOC


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


class PeakMemory:
    """Context manager recording peak CUDA allocation and completion status.

    The class form of :func:`measure_peak`, for sweeps that must survive an
    OOM and carry on to the next point::

        with PeakMemory(label=f"L={L}", suppress_oom=True) as pm:
            run_backward()
        rows.append(pm.as_dict())        # pm.completed says whether to trust it

    Attributes (valid after the block exits):
        peak_gib: peak PyTorch-allocated GiB, ``None`` without CUDA.  Excludes
            the ~0.8 GiB CUDA context / non-PyTorch overhead.
        reserved_gib: peak allocator-reserved GiB, ``None`` without CUDA.  A
            poor proxy for what can be served under
            ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
            (``docker/Dockerfile:6``); recorded, not trusted alone.
        completed: True iff the body raised nothing.
        oom: True iff the body raised a CUDA out-of-memory error.
        truncated: ``not completed`` -- see below.
        status: ``"completed"`` / ``"oom"`` / ``"error"`` (``"running"`` inside).
        error: ``repr(exc)`` if the body raised, else ``None``.
        wall_s: wall-clock seconds spent in the body.
        label: the caller's label, echoed back.

    Works with no torch and no CUDA: ``peak_gib`` / ``reserved_gib`` are
    ``None`` and everything else still reports honestly.
    """

    def __init__(
        self,
        device: Any = 0,
        reset: bool = True,
        suppress_oom: bool = False,
        label: str = "",
    ) -> None:
        self.device = device
        self.reset = reset
        self.suppress_oom = suppress_oom
        self.label = label
        self.peak_gib: float | None = None
        self.reserved_gib: float | None = None
        self.completed: bool = False
        self.oom: bool = False
        self.truncated: bool = True
        self.status: str = "pending"
        self.error: str | None = None
        self.wall_s: float | None = None
        self.cuda: bool = False
        self._t0: float | None = None

    def __enter__(self) -> "PeakMemory":
        self.cuda = has_cuda()
        self.status = "running"
        if self.reset:
            # See reset_peak: the counters are process-global and monotonic,
            # and pending kernels must land before the reset.
            _synchronize()
            reset_peak(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.wall_s = None if self._t0 is None else time.perf_counter() - self._t0
        if exc is None:
            self.completed = True
            self.oom = False
            self.truncated = False
            self.status = "completed"
            self.error = None
        else:
            self.completed = False
            self.oom = is_oom(exc)
            self.truncated = True
            self.status = "oom" if self.oom else "error"
            self.error = repr(exc)

        _synchronize()
        self.peak_gib = peak_allocated_gib(self.device)
        self.reserved_gib = peak_reserved_gib(self.device)
        log.info(
            "peak[%s] status=%s allocated=%s GiB reserved=%s GiB wall=%s s",
            self.label,
            self.status,
            _fmt(self.peak_gib),
            _fmt(self.reserved_gib),
            "n/a" if self.wall_s is None else f"{self.wall_s:.1f}",
        )
        # Swallow only an OOM, and only when asked: a sweep wants to record the
        # failure and move on, but any other error is a real bug.
        return bool(self.suppress_oom and self.oom)

    def as_dict(self) -> dict:
        """JSON-serialisable row.  ``truncated``/``completed`` are load-bearing."""
        return {
            "label": self.label,
            "status": self.status,
            "completed": self.completed,
            "oom": self.oom,
            "truncated": self.truncated,
            "peak_gib": self.peak_gib,
            "reserved_gib": self.reserved_gib,
            "wall_s": self.wall_s,
            "error": self.error,
            "cuda": self.cuda,
        }

    def __repr__(self) -> str:
        return (
            f"PeakMemory(label={self.label!r}, status={self.status!r}, "
            f"peak_gib={self.peak_gib}, truncated={self.truncated})"
        )


PeakMemory.__doc__ = (PeakMemory.__doc__ or "") + _TRUNCATED_DOC
