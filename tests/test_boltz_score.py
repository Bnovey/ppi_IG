"""Tests for igv.boltz_score checkpoint selection.

These are deliberately torch-free: select_checkpoint is the one part of
boltz_score that can run on CPU without the GPU container.
"""

import json
import sys
import warnings
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


# ---------------------------------------------------------------------------
# Chunk profile selection
#
# The pure part of the chunk-size logic: the >384-token branch is a different
# ALGORITHM from the <=384 one (small triangle chunks + all four MSA chunk knobs
# on, versus large chunks and every MSA knob off), which is why a size sweep has
# to be able to force one branch.
# ---------------------------------------------------------------------------

from igv.boltz_score import (  # noqa: E402
    _BOLTZ_TOKENS_2_2_1,
    _CHUNK_PROFILE_KEYS,
    _PROT_TOKEN_TO_LETTER_2_2_1,
    _build_yaml_sequences,
    _canonical_letter,
    _check_featurised_sequences,
    _residue_letter_table,
    chunk_profile,
    featurised_sequences,
    resolve_assert_feat_seq,
    resolve_autocast_dtype,
    resolve_msa_spec,
    resolve_tri_attn_ckpt,
    resolve_use_kernels,
    select_chunk_profile,
)

LARGE_EXPECTED = {
    "profile": "large",
    "pf_chunk": 128,
    "msa_chunk_heads_pwa": True,
    "msa_chunk_trans_z": 64,
    "msa_chunk_trans_msa": 32,
    "msa_chunk_outer": 4,
    "msa_chunk_tri": 128,
    "conf_chunk": 128,
}

SMALL_EXPECTED = {
    "profile": "small",
    "pf_chunk": 512,
    "msa_chunk_heads_pwa": False,
    "msa_chunk_trans_z": None,
    "msa_chunk_trans_msa": None,
    "msa_chunk_outer": None,
    "msa_chunk_tri": 512,
    "conf_chunk": 512,
}


def test_large_complex_gets_the_chunked_profile():
    """730 tokens (the 4fqi complex) must reproduce today's large-branch values."""
    assert select_chunk_profile(730, 384, {}) == LARGE_EXPECTED


def test_small_complex_gets_the_unchunked_profile():
    assert select_chunk_profile(230, 384, {}) == SMALL_EXPECTED


def test_profile_can_be_forced_large_below_the_threshold():
    """The property a memory/time sweep depends on.

    Without this, log(peak) vs log(L) is fitted across a point where the
    algorithm changes, and the exponent is meaningless.
    """
    forced = select_chunk_profile(230, 384, {"IGV_CHUNK_PROFILE": "large"})
    assert forced == LARGE_EXPECTED


def test_profile_can_be_forced_small_above_the_threshold():
    forced = select_chunk_profile(730, 384, {"IGV_CHUNK_PROFILE": "small"})
    assert forced == SMALL_EXPECTED


def test_pf_chunk_env_overrides_both_large_branch_chunks():
    got = select_chunk_profile(730, 384, {"IGV_PF_CHUNK": "32"})
    assert got["pf_chunk"] == 32
    assert got["msa_chunk_tri"] == 32


def test_pf_chunk_env_does_not_touch_msa_tri_in_the_small_profile():
    """Pre-existing asymmetry, preserved deliberately: in the small branch
    msa_chunk_tri is hardcoded 512 while pf_chunk honours the env var."""
    got = select_chunk_profile(230, 384, {"IGV_PF_CHUNK": "64"})
    assert got["pf_chunk"] == 64
    assert got["msa_chunk_tri"] == 512


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="IGV_CHUNK_PROFILE"):
        select_chunk_profile(730, 384, {"IGV_CHUNK_PROFILE": "bogus"})


def test_threshold_boundary_keeps_strictly_greater_semantics():
    """385 is large, 384 is small -- today's `n_tokens > threshold`."""
    assert select_chunk_profile(385, 384, {})["profile"] == "large"
    assert select_chunk_profile(384, 384, {})["profile"] == "small"


def test_chunk_profile_returns_exactly_the_contract_keys(monkeypatch):
    monkeypatch.delenv("IGV_CHUNK_PROFILE", raising=False)
    monkeypatch.delenv("IGV_PF_CHUNK", raising=False)
    got = chunk_profile(730)
    assert set(got) == set(_CHUNK_PROFILE_KEYS)
    assert got["pf_chunk"] == 128
    assert got["msa_chunk_trans_z"] == 64


def test_chunk_profile_argument_beats_size(monkeypatch):
    monkeypatch.delenv("IGV_CHUNK_PROFILE", raising=False)
    monkeypatch.delenv("IGV_PF_CHUNK", raising=False)
    assert chunk_profile(230, profile="large")["pf_chunk"] == 128
    assert chunk_profile(730, profile="small")["pf_chunk"] == 512


def test_chunk_profile_reads_env_when_no_argument(monkeypatch):
    monkeypatch.delenv("IGV_PF_CHUNK", raising=False)
    monkeypatch.setenv("IGV_CHUNK_PROFILE", "small")
    assert chunk_profile(730)["msa_chunk_heads_pwa"] is False


# ---------------------------------------------------------------------------
# IGV_AUTOCAST
# ---------------------------------------------------------------------------


def test_autocast_defaults_to_off_without_importing_torch():
    """Default must be None -- the repo has always run the trunk in fp32."""
    assert resolve_autocast_dtype(env={}) is None
    assert resolve_autocast_dtype(env={"IGV_AUTOCAST": "off"}) is None


def test_autocast_bf16_resolves_to_the_torch_dtype():
    torch = pytest.importorskip("torch")
    for spec in ("bf16", "bfloat16", "BF16", " bf16 "):
        assert resolve_autocast_dtype(env={"IGV_AUTOCAST": spec}) is torch.bfloat16


def test_autocast_rejects_fp16_rather_than_warning():
    """fp16 must RAISE: there is no GradScaler in this repo, and its 5-bit
    exponent breaks the finiteness / non-zero guards after the checkpoint."""
    for spec in ("fp16", "float16", "half"):
        with pytest.raises(ValueError, match="GradScaler"):
            resolve_autocast_dtype(env={"IGV_AUTOCAST": spec})


def test_autocast_rejects_garbage():
    with pytest.raises(ValueError, match="IGV_AUTOCAST"):
        resolve_autocast_dtype(env={"IGV_AUTOCAST": "tf32ish"})


def test_autocast_explicit_argument_beats_env():
    assert resolve_autocast_dtype("off", env={"IGV_AUTOCAST": "bf16"}) is None


# ---------------------------------------------------------------------------
# IGV_USE_KERNELS / IGV_TRI_ATTN_KERNEL / IGV_TRI_ATTN_CKPT
# ---------------------------------------------------------------------------


def _cueq_available() -> bool:
    try:
        import cuequivariance_torch  # noqa: F401
    except ImportError:
        return False
    return True


def test_use_kernels_defaults_to_false():
    assert resolve_use_kernels(env={}) == (False, False)


def test_use_kernels_does_not_inherit_the_boltz_model_flag():
    """The eight call sites hardcoded `False` before IGV_USE_KERNELS existed.

    model.use_kernels is restored from the checkpoint's hyperparameters, so
    inheriting it would let a checkpoint silently switch the trunk to the kernel
    path (which also disables triangle-attention chunking) on a run where
    nothing was set -- and then hard-fail on the banned cuequivariance_torch.
    """

    class _Off:
        use_kernels = False

    class _On:
        use_kernels = True

    assert resolve_use_kernels(_Off(), env={}) == (False, False)
    # The load-bearing case: True on the model must NOT turn kernels on, and
    # must not raise the cuequivariance error either.
    assert resolve_use_kernels(_On(), env={}) == (False, False)
    # An explicit request still wins over the default.
    assert resolve_use_kernels(_Off(), spec="0", env={}) == (False, False)


def test_use_kernels_fails_eagerly_when_cuequivariance_is_banned():
    """The flag must fail here, not from inside a backward recompute stack."""
    if _cueq_available():  # pragma: no cover - not this environment
        assert resolve_use_kernels(env={"IGV_USE_KERNELS": "1"}) == (True, False)
        return
    with pytest.raises(RuntimeError, match="constraints.txt"):
        resolve_use_kernels(env={"IGV_USE_KERNELS": "1"})
    with pytest.raises(RuntimeError, match="cuequivariance_torch"):
        resolve_use_kernels(env={"IGV_TRI_ATTN_KERNEL": "1"})


def test_use_kernels_rejects_non_boolean_env():
    with pytest.raises(ValueError, match="IGV_USE_KERNELS"):
        resolve_use_kernels(env={"IGV_USE_KERNELS": "maybe"})


def test_tri_attn_ckpt_defaults_off():
    assert resolve_tri_attn_ckpt(env={}) is False
    assert resolve_tri_attn_ckpt(env={"IGV_TRI_ATTN_CKPT": "0"}) is False


def test_tri_attn_ckpt_env_and_argument():
    assert resolve_tri_attn_ckpt(env={"IGV_TRI_ATTN_CKPT": "1"}) is True
    assert resolve_tri_attn_ckpt(env={"IGV_TRI_ATTN_CKPT": "yes"}) is True
    # An explicit argument wins over the env, like the other knobs.
    assert resolve_tri_attn_ckpt(False, env={"IGV_TRI_ATTN_CKPT": "1"}) is False
    with pytest.raises(ValueError, match="IGV_TRI_ATTN_CKPT"):
        resolve_tri_attn_ckpt(env={"IGV_TRI_ATTN_CKPT": "sometimes"})


def test_msa_spec_defaults_to_none():
    assert resolve_msa_spec(env={}) is None
    assert resolve_msa_spec(env={"IGV_MSA_SPEC": ""}) is None
    assert resolve_msa_spec(env={"IGV_MSA_SPEC": "empty"}) == "empty"
    assert resolve_msa_spec("empty", env={"IGV_MSA_SPEC": "x.a3m"}) == "empty"


def test_assert_feat_seq_defaults_on():
    assert resolve_assert_feat_seq(env={}) is True
    assert resolve_assert_feat_seq(env={"IGV_ASSERT_FEAT_SEQ": "0"}) is False


# ---------------------------------------------------------------------------
# Featurised-sequence decoding
#
# This is the guard the token-COUNT assertion cannot be: every mutant in the
# 4fqi H1 library is same-length, so a stale Boltz cache would pass the count
# check silently.
# ---------------------------------------------------------------------------

_LETTER_TO_TOKEN = {v: k for k, v in _PROT_TOKEN_TO_LETTER_2_2_1.items()}


def _synthetic_feats(chains_letters, n_pad: int = 0):
    """Build a minimal feats dict: one-hot res_type, asym_id runs, pad mask."""
    import torch

    ids, asym = [], []
    for asym_val, letters in enumerate(chains_letters):
        for letter in letters:
            ids.append(_BOLTZ_TOKENS_2_2_1.index(_LETTER_TO_TOKEN[letter]))
            asym.append(asym_val)
    keep = [1] * len(ids)
    for _ in range(n_pad):
        ids.append(0)  # <pad>
        asym.append(len(chains_letters))
        keep.append(0)

    res_type = torch.nn.functional.one_hot(
        torch.tensor(ids), num_classes=len(_BOLTZ_TOKENS_2_2_1)
    ).unsqueeze(0)
    return {
        "res_type": res_type,
        "asym_id": torch.tensor(asym).unsqueeze(0),
        "token_pad_mask": torch.tensor(keep).unsqueeze(0),
    }


def test_letter_table_uses_the_pinned_fallback_without_boltz():
    """boltz is not installed in CI, so the fallback table must carry it."""
    table = _residue_letter_table()
    assert len(table) == len(_BOLTZ_TOKENS_2_2_1) == 33
    assert table[_BOLTZ_TOKENS_2_2_1.index("ALA")] == "A"
    assert table[_BOLTZ_TOKENS_2_2_1.index("TRP")] == "W"
    assert table[_BOLTZ_TOKENS_2_2_1.index("UNK")] == "X"


def test_featurised_sequences_decodes_two_chains():
    pytest.importorskip("torch")
    feats = _synthetic_feats(["ACD", "EF"])
    assert featurised_sequences(feats) == {0: "ACD", 1: "EF"}


def test_featurised_sequences_excludes_padding():
    pytest.importorskip("torch")
    assert featurised_sequences(_synthetic_feats(["ACD", "EF"], n_pad=2)) == {
        0: "ACD",
        1: "EF",
    }


def test_featurised_sequences_round_trips_unk_as_x():
    pytest.importorskip("torch")
    assert featurised_sequences(_synthetic_feats(["AXC"])) == {0: "AXC"}


def test_sequence_check_passes_on_matching_features(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("IGV_ASSERT_FEAT_SEQ", raising=False)
    chains = {"H": "ACD", "L": "EF"}
    feats = _synthetic_feats(["ACD", "EF"])
    token_map = {("H", 0): 0, ("H", 1): 1, ("H", 2): 2, ("L", 0): 3, ("L", 1): 4}
    _check_featurised_sequences(chains, feats, token_map)  # must not raise


def test_sequence_check_catches_a_stale_cache(monkeypatch):
    """A same-length mutation that the token-count assertion cannot see."""
    pytest.importorskip("torch")
    monkeypatch.delenv("IGV_ASSERT_FEAT_SEQ", raising=False)
    chains = {"H": "ACD"}          # what we asked for
    feats = _synthetic_feats(["ACE"])  # what a reused cache_dir returned
    token_map = {("H", 0): 0, ("H", 1): 1, ("H", 2): 2}
    with pytest.raises(RuntimeError) as excinfo:
        _check_featurised_sequences(chains, feats, token_map)
    msg = str(excinfo.value)
    assert "residue index 2" in msg
    assert "'D'" in msg and "'E'" in msg
    # The message must name the actual cause, not just the symptom.
    assert "process_inputs" in msg and "cache_dir" in msg


def test_sequence_check_can_be_downgraded_for_debugging(monkeypatch, caplog):
    pytest.importorskip("torch")
    monkeypatch.setenv("IGV_ASSERT_FEAT_SEQ", "0")
    chains = {"H": "ACD"}
    feats = _synthetic_feats(["ACE"])
    token_map = {("H", 0): 0, ("H", 1): 1, ("H", 2): 2}
    with caplog.at_level("ERROR"):
        _check_featurised_sequences(chains, feats, token_map)
    assert any("Featurised sequence does not match" in r.message for r in caplog.records)


def test_ambiguous_letters_canonicalise_the_way_boltz_does():
    """B/J/Z/O/U/X all become UNK and decode back to "X".

    Comparing raw letters instead would fire a spurious "stale cache" error on
    any sequence containing one of them.
    """
    assert _canonical_letter("A") == "A"
    assert _canonical_letter("X") == "X"
    for letter in ("B", "J", "Z", "O", "U"):
        assert _canonical_letter(letter) == "X"


def test_sequence_check_tolerates_unmappable_requested_letters(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("IGV_ASSERT_FEAT_SEQ", raising=False)
    chains = {"H": "ABC"}                # 'B' is not a canonical residue
    feats = _synthetic_feats(["AXC"])    # boltz featurises it as UNK -> 'X'
    token_map = {("H", 0): 0, ("H", 1): 1, ("H", 2): 2}
    _check_featurised_sequences(chains, feats, token_map)  # must not raise


# ---------------------------------------------------------------------------
# IGV_TRI_ATTN_CKPT: the rebind itself
#
# The memory effect needs boltz + a CUDA backward, but the part that actually
# breaks -- the keyword contract between chunk_layer and the wrapper, and
# whether the rebind takes effect at all -- is testable with a stand-in for
# boltz's TriangleAttention. chunk_layer calls `layer(**chunks)`, so a wrapper
# with the wrong parameter names fails here instead of on the GPU.
# ---------------------------------------------------------------------------


def _install_fake_boltz_triangle_attention(monkeypatch):
    import types as _types

    import torch
    import torch.nn as nn

    modules = {}

    def _mk(name):
        mod = _types.ModuleType(name)
        modules[name] = mod
        return mod

    for name in (
        "boltz",
        "boltz.model",
        "boltz.model.layers",
        "boltz.model.layers.triangular_attention",
    ):
        _mk(name)
    attention_mod = _mk("boltz.model.layers.triangular_attention.attention")
    utils_mod = _mk("boltz.model.layers.triangular_attention.utils")

    class TriangleAttention(nn.Module):
        """Stand-in with boltz's `mha` / `_chunk` interface."""

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2, bias=False)
            self.mha_calls = []

        def mha(self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=False):
            self.mha_calls.append(use_kernels)
            return self.lin(q_x) + tri_bias

        def _chunk(self, x, tri_bias, mask_bias, mask, chunk_size, use_kernels=False):
            raise AssertionError("the original _chunk must have been rebound")

    def chunk_layer(layer, inputs, chunk_size=None, no_batch_dims=0, _out=None):
        """Mirrors boltz chunk_layer's contract: keyword call, no no_grad."""
        n = inputs["q_x"].shape[0]
        return torch.cat(
            [
                layer(**{k: v[i : i + chunk_size] for k, v in inputs.items()})
                for i in range(0, n, chunk_size)
            ],
            dim=0,
        )

    attention_mod.TriangleAttention = TriangleAttention
    utils_mod.chunk_layer = chunk_layer
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return TriangleAttention


def test_tri_attn_chunk_checkpointing_rebinds_and_recomputes(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from igv.boltz_score import enable_triangle_attention_chunk_checkpointing

    TriangleAttention = _install_fake_boltz_triangle_attention(monkeypatch)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TriangleAttention()

    model = _Model()
    assert enable_triangle_attention_chunk_checkpointing(model) == 1
    # Idempotent: the sentinel stops a second wrap of the same module.
    assert enable_triangle_attention_chunk_checkpointing(model) == 0

    x = torch.randn(4, 3, 2, requires_grad=True)
    bias = torch.zeros(4, 3, 2)
    out = model.attn._chunk(x, bias, bias, bias, 2)
    assert out.shape == (4, 3, 2)
    n_forward_calls = len(model.attn.mha_calls)
    assert n_forward_calls == 2  # two chunks of two

    out.sum().backward()
    # The saving comes from recompute: mha runs again during backward.
    assert len(model.attn.mha_calls) > n_forward_calls
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_numerics_arm_records_what_changes_the_numbers():
    """An artifact must be able to say which dtype produced it.

    Measured on igv-gpu 2026-09-04 at L=730: bf16 changes grad_abs_max by up to
    3.6% versus fp32. Before this helper, stages 02/03/04/07 stamped an arm dict
    with no dtype in it, so a bf16 artifact and an fp32 artifact were
    indistinguishable in provenance and silently comparable.
    """
    from igv.boltz_score import numerics_arm

    off = numerics_arm({})
    assert off["autocast"] == "off"
    assert off["dtype"] == "fp32"
    assert off["tri_attn_ckpt"] is False
    assert off["pf_group_size"] == 4
    assert off["use_kernels"] is False

    on = numerics_arm(
        {
            "IGV_AUTOCAST": "bf16",
            "IGV_TRI_ATTN_CKPT": "1",
            "IGV_PF_GROUP_SIZE": "8",
            "IGV_PF_CHUNK": "16",
            "IGV_CHUNK_PROFILE": "large",
        }
    )
    assert on["dtype"] == "bf16"
    assert on["tri_attn_ckpt"] is True
    assert on["pf_group_size"] == 8
    assert on["pf_chunk"] == "16"
    assert on["chunk_profile"] == "large"

    # The whole point: the two are distinguishable.
    assert off != on
    # JSON-safe -- it goes straight into a provenance sidecar, and a torch
    # dtype object would not serialise.
    json.dumps(off)
    json.dumps(on)
    # A group size of 0 or negative would silently disable grouping; clamped.
    assert numerics_arm({"IGV_PF_GROUP_SIZE": "0"})["pf_group_size"] == 1


def test_count_tri_attn_chunk_ckpt_survives_model_reuse(monkeypatch):
    """"0 newly wrapped" must not be readable as "the lever is off".

    Observed on igv-gpu: 08_memscale wrapped 156 modules at its first ladder
    point and 0 at every later one, because it reuses one model object and the
    rebind is idempotent. The lever was fully active throughout, but the log
    said "Expect no memory change" -- which would have invalidated a reader's
    interpretation of the whole sweep. The instrumented count is the honest
    signal, and it is a property of the model, not of the last call.
    """
    pytest.importorskip("torch")
    import torch.nn as nn

    from igv.boltz_score import (
        count_tri_attn_chunk_ckpt,
        enable_triangle_attention_chunk_checkpointing,
    )

    TriangleAttention = _install_fake_boltz_triangle_attention(monkeypatch)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = TriangleAttention()
            self.b = TriangleAttention()

    model = _Model()
    assert count_tri_attn_chunk_ckpt(model) == 0
    assert enable_triangle_attention_chunk_checkpointing(model) == 2
    assert count_tri_attn_chunk_ckpt(model) == 2
    # The reuse case: nothing new to wrap, still fully instrumented.
    assert enable_triangle_attention_chunk_checkpointing(model) == 0
    assert count_tri_attn_chunk_ckpt(model) == 2
    # A model with no TriangleAttention at all is the case that IS inert.
    assert count_tri_attn_chunk_ckpt(nn.Linear(2, 2)) == 0


def test_tri_attn_chunk_checkpointing_is_numerically_transparent(monkeypatch):
    """Checkpointing is a memory/compute trade, not a numerical change."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from igv.boltz_score import enable_triangle_attention_chunk_checkpointing

    TriangleAttention = _install_fake_boltz_triangle_attention(monkeypatch)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TriangleAttention()

    plain = TriangleAttention()
    model = _Model()
    model.attn.load_state_dict(plain.state_dict())
    enable_triangle_attention_chunk_checkpointing(model)

    x = torch.randn(4, 3, 2, requires_grad=True)
    bias = torch.zeros(4, 3, 2)
    reference = plain.mha(x, x, bias, bias, bias, False)
    got = model.attn._chunk(x, bias, bias, bias, 2)
    assert torch.equal(got, reference)


def test_tri_attn_chunk_checkpointing_bypasses_when_grad_is_off(monkeypatch):
    """Under the outer reentrant checkpoint's no_grad forward, a non-reentrant
    checkpoint would warn once per chunk per attention per layer."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from igv.boltz_score import enable_triangle_attention_chunk_checkpointing

    TriangleAttention = _install_fake_boltz_triangle_attention(monkeypatch)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TriangleAttention()

    model = _Model()
    enable_triangle_attention_chunk_checkpointing(model)
    x = torch.randn(4, 3, 2)
    bias = torch.zeros(4, 3, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any checkpoint warning fails the test
        with torch.no_grad():
            out = model.attn._chunk(x, bias, bias, bias, 2)
    assert out.shape == (4, 3, 2)
    assert not out.requires_grad


def test_msa_server_resolves_to_no_yaml_key():
    """"server" means "let the MSA server generate one", i.e. emit no key.

    Emitting the literal string would make boltz treat it as a path to a custom
    .a3m/.csv. scripts/08_memscale.py passes this value by name.
    """
    assert resolve_msa_spec("server", env={}) is None
    assert resolve_msa_spec(env={"IGV_MSA_SPEC": "server"}) is None


def test_build_complex_feats_accepts_the_msa_spec_alias():
    import inspect

    from igv.boltz_score import build_complex_feats

    params = inspect.signature(build_complex_feats).parameters
    assert "msa" in params and "msa_spec" in params
    assert params["msa"].default is None and params["msa_spec"].default is None


# ---------------------------------------------------------------------------
# _build_yaml_sequences: per-chain MSA dict support
# ---------------------------------------------------------------------------


def test_yaml_sequences_none_emits_no_msa_key():
    chains = {"H": "ACD", "L": "EF"}
    seqs = _build_yaml_sequences(chains, None)
    assert len(seqs) == 2
    assert seqs[0] == {"protein": {"id": "H", "sequence": "ACD"}}
    assert seqs[1] == {"protein": {"id": "L", "sequence": "EF"}}
    for s in seqs:
        assert "msa" not in s["protein"]


def test_yaml_sequences_string_emits_same_msa_for_all_chains():
    chains = {"H": "ACD", "L": "EF"}
    seqs = _build_yaml_sequences(chains, "empty")
    assert seqs[0]["protein"]["msa"] == "empty"
    assert seqs[1]["protein"]["msa"] == "empty"


def test_yaml_sequences_dict_emits_per_chain_msa(tmp_path):
    (tmp_path / "h.csv").write_text("header\ndata")
    (tmp_path / "l.csv").write_text("header\ndata")
    chains = {"H": "ACD", "L": "EF"}
    msa_dict = {"H": tmp_path / "h.csv", "L": tmp_path / "l.csv"}
    seqs = _build_yaml_sequences(chains, msa_dict)
    assert seqs[0]["protein"]["msa"] == str(tmp_path / "h.csv")
    assert seqs[1]["protein"]["msa"] == str(tmp_path / "l.csv")


def test_yaml_sequences_dict_preserves_chain_order(tmp_path):
    for name in ("a.csv", "b.csv", "c.csv"):
        (tmp_path / name).write_text("data")
    chains = {"Z": "A", "M": "CD", "A": "EFG"}
    msa_dict = {
        "Z": tmp_path / "a.csv",
        "M": tmp_path / "b.csv",
        "A": tmp_path / "c.csv",
    }
    seqs = _build_yaml_sequences(chains, msa_dict)
    assert [s["protein"]["id"] for s in seqs] == ["Z", "M", "A"]
    assert seqs[0]["protein"]["msa"] == str(tmp_path / "a.csv")
    assert seqs[2]["protein"]["msa"] == str(tmp_path / "c.csv")


def test_yaml_sequences_dict_missing_chain_raises(tmp_path):
    (tmp_path / "h.csv").write_text("data")
    chains = {"H": "ACD", "L": "EF"}
    msa_dict = {"H": tmp_path / "h.csv"}
    with pytest.raises(ValueError, match="missing chains.*L"):
        _build_yaml_sequences(chains, msa_dict)


def test_yaml_sequences_dict_nonexistent_path_raises(tmp_path):
    chains = {"H": "ACD"}
    msa_dict = {"H": tmp_path / "does_not_exist.csv"}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _build_yaml_sequences(chains, msa_dict)


def test_resolve_msa_spec_passes_dict_through():
    d = {"H": "/some/path.csv", "L": "/other/path.csv"}
    assert resolve_msa_spec(d) is d


def test_resolve_msa_spec_none_server_empty_string_unchanged():
    assert resolve_msa_spec(None, env={}) is None
    assert resolve_msa_spec("server", env={}) is None
    assert resolve_msa_spec("empty", env={}) == "empty"
    assert resolve_msa_spec("/path/to/file.a3m", env={}) == "/path/to/file.a3m"


def test_msa_spec_alias_conflict_still_fires_with_dict():
    """Passing both msa and msa_spec with different values must still raise."""
    from igv.boltz_score import build_complex_feats
    import inspect

    sig = inspect.signature(build_complex_feats)
    assert "msa" in sig.parameters
    assert "msa_spec" in sig.parameters


# ---------------------------------------------------------------------------
# MSA file to chain-id mapping (stage 02 logic)
# ---------------------------------------------------------------------------


def _create_mock_msa_files(msa_dir: Path, count: int) -> list[Path]:
    """Create input_0.csv, input_1.csv, ... in *msa_dir*."""
    msa_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(count):
        f = msa_dir / f"input_{i}.csv"
        f.write_text(f"mock_msa_{i}")
        files.append(f)
    return files


def test_msa_file_to_chain_mapping_matches_chain_order(tmp_path):
    """The mapping from input_N.csv -> chain id must follow chains.items() order."""
    chains = {"H": "EVQL", "L": "DIQM", "A": "MKYL"}
    msa_dir = tmp_path / "msa"
    _create_mock_msa_files(msa_dir, 3)

    msa_files = sorted(msa_dir.glob("input_*.csv"))
    chain_ids = list(chains.keys())

    assert len(msa_files) == len(chain_ids)
    mapping = dict(zip(chain_ids, msa_files))
    assert list(mapping.keys()) == ["H", "L", "A"]
    assert mapping["H"].name == "input_0.csv"
    assert mapping["L"].name == "input_1.csv"
    assert mapping["A"].name == "input_2.csv"


def test_msa_file_count_mismatch_detected(tmp_path):
    """If MSA file count != chain count, the stage must fail, not guess."""
    chains = {"H": "EVQL", "L": "DIQM", "A": "MKYL"}
    msa_dir = tmp_path / "msa"
    _create_mock_msa_files(msa_dir, 2)  # only 2 files for 3 chains

    msa_files = sorted(msa_dir.glob("input_*.csv"))
    chain_ids = list(chains.keys())

    assert len(msa_files) != len(chain_ids)


def test_msa_file_count_mismatch_raises_clear_error(tmp_path):
    """Simulate the count-mismatch error path from stage 02."""
    chains = {"H": "EVQL", "L": "DIQM"}
    msa_dir = tmp_path / "msa"
    _create_mock_msa_files(msa_dir, 3)  # 3 files for 2 chains

    msa_files = sorted(msa_dir.glob("input_*.csv"))
    chain_ids = list(chains.keys())

    with pytest.raises(RuntimeError, match="Cannot build per-chain MSA mapping"):
        if len(msa_files) != len(chain_ids):
            raise RuntimeError(
                f"Expected {len(chain_ids)} MSA files in {msa_dir} "
                f"(one per chain: {chain_ids}), found {len(msa_files)}: "
                f"{[f.name for f in msa_files]}. Cannot build per-chain MSA mapping."
            )


# ---------------------------------------------------------------------------
# compute_homopolymer_embeddings: cache dir uniqueness
# ---------------------------------------------------------------------------


def test_homopolymer_cache_dirs_are_distinct_per_amino_acid(monkeypatch, tmp_path):
    """Each of the 20 homopolymer featurisations must get a unique cache dir."""
    from unittest.mock import MagicMock, patch

    from igv.attrib import CANONICAL_AMINO_ACIDS
    from igv.boltz_score import compute_homopolymer_embeddings

    torch = pytest.importorskip("torch")

    seen_cache_dirs = []
    seen_chains = []

    def mock_build_complex_feats(chains, structure_pdb, cache_dir, device, **kwargs):
        seen_cache_dirs.append(str(cache_dir))
        seen_chains.append(dict(chains))
        token_map = {}
        offset = 0
        for cid, seq in chains.items():
            for i in range(len(seq)):
                token_map[(cid, i)] = offset + i
            offset += len(seq)
        feats = {"dummy": True}
        return feats, token_map

    def mock_embedder_only(model, feats):
        return torch.randn(1, 15, 32)

    with patch("igv.boltz_score.build_complex_feats", side_effect=mock_build_complex_feats):
        with patch("igv.boltz_score.embedder_only", side_effect=mock_embedder_only):
            result = compute_homopolymer_embeddings(
                model=MagicMock(),
                chains={"H": "ACDEF", "L": "GHIKLMNPQR"},
                chain="H",
                structure_pdb=tmp_path / "dummy.pdb",
                cache_dir=tmp_path,
                device="cpu",
            )

    assert len(result) == len(CANONICAL_AMINO_ACIDS)
    assert len(seen_cache_dirs) == 20
    assert len(set(seen_cache_dirs)) == 20, (
        f"Cache dirs must be unique per amino acid, got duplicates: {seen_cache_dirs}"
    )
    for aa, cache_path in zip(CANONICAL_AMINO_ACIDS, seen_cache_dirs):
        assert aa in cache_path, f"Cache dir for {aa} should contain the AA letter"

    for aa, chains_used in zip(CANONICAL_AMINO_ACIDS, seen_chains):
        assert chains_used["H"] == aa * 5, (
            f"Chain H should be homopolymer of {aa}, got {chains_used['H']!r}"
        )
        assert chains_used["L"] == "GHIKLMNPQR"


def test_homopolymer_embeddings_use_empty_msa(monkeypatch, tmp_path):
    """Homopolymer featurisations must use msa='empty'."""
    from unittest.mock import MagicMock, patch

    torch = pytest.importorskip("torch")

    seen_msa = []

    def mock_build_complex_feats(chains, structure_pdb, cache_dir, device, **kwargs):
        seen_msa.append(kwargs.get("msa"))
        token_map = {}
        offset = 0
        for cid, seq in chains.items():
            for i in range(len(seq)):
                token_map[(cid, i)] = offset + i
            offset += len(seq)
        return {"dummy": True}, token_map

    def mock_embedder_only(model, feats):
        return torch.randn(1, 8, 16)

    with patch("igv.boltz_score.build_complex_feats", side_effect=mock_build_complex_feats):
        with patch("igv.boltz_score.embedder_only", side_effect=mock_embedder_only):
            from igv.boltz_score import compute_homopolymer_embeddings

            compute_homopolymer_embeddings(
                model=MagicMock(),
                chains={"A": "ACD", "B": "EFGHI"},
                chain="A",
                structure_pdb=tmp_path / "dummy.pdb",
                cache_dir=tmp_path,
                device="cpu",
            )

    assert all(m == "empty" for m in seen_msa), (
        f"All homopolymer featurisations must use msa='empty', got {seen_msa}"
    )


def test_homopolymer_embedding_shape_matches_baseline_contract(tmp_path):
    """Shape of returned tensors must match build_mean_aa_baseline's contract."""
    from unittest.mock import MagicMock, patch

    torch = pytest.importorskip("torch")

    from igv.attrib import CANONICAL_AMINO_ACIDS, build_mean_aa_baseline
    from igv.boltz_score import compute_homopolymer_embeddings

    D = 24
    chain_len = 7
    total_tokens = 15

    def mock_build_complex_feats(chains, structure_pdb, cache_dir, device, **kwargs):
        token_map = {}
        offset = 0
        for cid, seq in chains.items():
            for i in range(len(seq)):
                token_map[(cid, i)] = offset + i
            offset += len(seq)
        return {"dummy": True}, token_map

    def mock_embedder_only(model, feats):
        return torch.randn(1, total_tokens, D)

    with patch("igv.boltz_score.build_complex_feats", side_effect=mock_build_complex_feats):
        with patch("igv.boltz_score.embedder_only", side_effect=mock_embedder_only):
            per_aa = compute_homopolymer_embeddings(
                model=MagicMock(),
                chains={"H": "A" * chain_len, "L": "G" * (total_tokens - chain_len)},
                chain="H",
                structure_pdb=tmp_path / "dummy.pdb",
                cache_dir=tmp_path,
                device="cpu",
            )

    assert len(per_aa) == len(CANONICAL_AMINO_ACIDS)
    for t in per_aa:
        assert t.shape == (chain_len, D)

    import numpy as np

    embeddings = torch.randn(1, total_tokens, D)
    peptide_idx = np.arange(chain_len)
    baseline = build_mean_aa_baseline(embeddings, peptide_idx, per_aa)
    assert baseline.shape == embeddings.shape


# ---------------------------------------------------------------------------
# --baseline zeros is unchanged
# ---------------------------------------------------------------------------


def test_baseline_zeros_unchanged():
    """The zeros baseline must be bit-identical to torch.zeros_like."""
    torch = pytest.importorskip("torch")
    s = torch.randn(1, 20, 64)
    baseline = torch.zeros_like(s)
    assert torch.equal(baseline, torch.zeros_like(s))
