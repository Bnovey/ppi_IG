"""Tests for scripts/15_pair_capture.py -- all CPU, no boltz.

Exercises:
  - Gate B concentration statistics on synthetic maps with known answers
  - Token map serialisation round-trip
  - Ladder plumbing and .npz key layout
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest
import torch

from igv.attrib import pair_layer_ig

# Import from the script
from importlib import import_module
pair_capture = import_module("15_pair_capture")
gate_b_statistics = pair_capture.gate_b_statistics
serialise_token_map = pair_capture.serialise_token_map
deserialise_token_map = pair_capture.deserialise_token_map


# ---------------------------------------------------------------------------
# 1. Gate B: flat map detection
# ---------------------------------------------------------------------------


class TestGateBFlatMap:
    """A deliberately flat (uniform) map should show:
    - CV close to 0
    - near_constant = True
    - enrichment close to 1.0x for all k
    - diagonal and off-diagonal shares close to their geometric expectations
    """

    def test_constant_map(self):
        L = 50
        pair_map = np.ones((L, L), dtype=np.float32) * 3.14
        stats = gate_b_statistics(pair_map)

        assert stats["near_constant"] is True
        assert stats["cv_std_over_mean"] < 0.01
        assert stats["zero_fraction"] == 0.0
        for k in [10, 100, 1000]:
            if stats.get(f"top_{k}_enrichment") is not None:
                assert abs(stats[f"top_{k}_enrichment"] - 1.0) < 0.05

    def test_zero_map(self):
        L = 20
        pair_map = np.zeros((L, L), dtype=np.float32)
        stats = gate_b_statistics(pair_map)

        assert stats["zero_fraction"] == 1.0
        assert stats["mean_abs"] == 0.0


# ---------------------------------------------------------------------------
# 2. Gate B: diagonal-only map detection
# ---------------------------------------------------------------------------


class TestGateBDiagonalMap:
    """A diagonal-only map should show:
    - diag_mass_share = 1.0
    - off_diag_mass_share = 0.0
    - all upper-triangle entries are zero
    """

    def test_diagonal_only(self):
        L = 30
        pair_map = np.zeros((L, L), dtype=np.float32)
        np.fill_diagonal(pair_map, np.arange(1, L + 1, dtype=np.float32))
        stats = gate_b_statistics(pair_map)

        assert stats["diag_mass_share"] == pytest.approx(1.0)
        assert stats["off_diag_mass_share"] == pytest.approx(0.0)
        assert stats["mean_abs"] == 0.0


# ---------------------------------------------------------------------------
# 3. Gate B: single-hot-row map detection
# ---------------------------------------------------------------------------


class TestGateBSingleHotRow:
    """A map concentrated on one row should show:
    - max_single_row_mass_share close to 1.0
    """

    def test_one_row_dominates(self):
        L = 40
        pair_map = np.zeros((L, L), dtype=np.float32)
        pair_map[5, :] = 10.0
        pair_map[:, 5] = 10.0
        stats = gate_b_statistics(pair_map)

        assert stats["max_single_row_mass_share"] > 0.45
        assert stats["max_single_row_index"] == 5

    def test_pure_single_row(self):
        L = 40
        pair_map = np.zeros((L, L), dtype=np.float32)
        pair_map[3, :] = 10.0
        stats = gate_b_statistics(pair_map)

        assert stats["max_single_row_mass_share"] > 0.95
        assert stats["max_single_row_index"] == 3

    def test_uniform_rows_low_max_share(self):
        L = 50
        rng = np.random.default_rng(42)
        pair_map = rng.normal(0, 1, (L, L)).astype(np.float32)
        pair_map = (pair_map + pair_map.T) / 2
        stats = gate_b_statistics(pair_map)

        assert stats["max_single_row_mass_share"] < 0.1


# ---------------------------------------------------------------------------
# 4. Gate B: enrichment on a concentrated map
# ---------------------------------------------------------------------------


class TestGateBConcentrated:
    """A map with a few very large entries should show high enrichment."""

    def test_top_k_enrichment(self):
        L = 100
        pair_map = np.ones((L, L), dtype=np.float32) * 0.001
        pair_map[0, 1] = 100.0
        pair_map[1, 0] = 100.0
        pair_map[2, 3] = 50.0
        pair_map[3, 2] = 50.0
        stats = gate_b_statistics(pair_map)

        assert stats["top_10_enrichment"] > 5.0
        assert not stats["near_constant"]


# ---------------------------------------------------------------------------
# 5. Token map round-trip
# ---------------------------------------------------------------------------


class TestTokenMapRoundTrip:

    def test_round_trip(self):
        chains = {"A": "ACDEFG", "B": "HIKLM"}
        token_map = {
            ("A", 0): 0, ("A", 1): 1, ("A", 2): 2,
            ("A", 3): 3, ("A", 4): 4, ("A", 5): 5,
            ("B", 0): 6, ("B", 1): 7, ("B", 2): 8,
            ("B", 3): 9, ("B", 4): 10,
        }
        arrays = serialise_token_map(token_map, chains)

        assert "token_map_keys" in arrays
        assert "token_map_values" in arrays
        assert "chain_order" in arrays
        assert "chain_offsets" in arrays
        assert "chain_lengths" in arrays

        assert list(arrays["chain_order"]) == ["A", "B"]
        np.testing.assert_array_equal(arrays["chain_offsets"], [0, 6])
        np.testing.assert_array_equal(arrays["chain_lengths"], [6, 5])

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            np.savez(tmp.name, **arrays)
            loaded = np.load(tmp.name, allow_pickle=True)
            recovered = deserialise_token_map(loaded)

        assert recovered == token_map

    def test_chain_order_preserved(self):
        chains = {"B": "ABC", "A": "DEFG"}
        token_map = {
            ("B", 0): 0, ("B", 1): 1, ("B", 2): 2,
            ("A", 0): 3, ("A", 1): 4, ("A", 2): 5, ("A", 3): 6,
        }
        arrays = serialise_token_map(token_map, chains)
        assert list(arrays["chain_order"]) == ["B", "A"]


# ---------------------------------------------------------------------------
# 6. .npz key layout contract
# ---------------------------------------------------------------------------


class TestNpzKeyLayout:
    """Verify the .npz produced by a synthetic run has all required keys."""

    def test_expected_keys_present(self):
        L, C = 10, 128
        ladder = [8, 16, 32, 64]

        rung_maps = {}
        for m in ladder:
            z_x = torch.randn(1, L, L, C, dtype=torch.float64)
            z_b = torch.zeros(1, L, L, C, dtype=torch.float64)
            result = pair_layer_ig(lambda z: z.sum(), z_b, z_x, m_steps=m)
            rung_maps[m] = result.interaction_map.detach().cpu().numpy().astype(np.float32)

        final_result = pair_layer_ig(
            lambda z: z.sum(),
            torch.zeros(1, L, L, C, dtype=torch.float64),
            torch.randn(1, L, L, C, dtype=torch.float64),
            m_steps=64,
        )

        chains = {"A": "A" * 6, "B": "B" * 4}
        token_map = {("A", i): i for i in range(6)}
        token_map.update({("B", i): 6 + i for i in range(4)})

        save_dict = {"pair_ig": rung_maps[64]}
        for m in ladder:
            save_dict[f"pair_ig_m{m}"] = rung_maps[m]
        save_dict["grad_m64"] = final_result.grad.detach().cpu().numpy().astype(np.float32)
        save_dict.update(serialise_token_map(token_map, chains))

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            np.savez(tmp.name, **save_dict)
            loaded = np.load(tmp.name, allow_pickle=True)
            keys = set(loaded.keys())

        required = {
            "pair_ig",
            "pair_ig_m8", "pair_ig_m16", "pair_ig_m32", "pair_ig_m64",
            "grad_m64",
            "token_map_keys", "token_map_values",
            "chain_order", "chain_offsets", "chain_lengths",
        }
        assert required.issubset(keys), f"Missing keys: {required - keys}"


# ---------------------------------------------------------------------------
# 7. Ladder plumbing: convergence Spearman on identical maps
# ---------------------------------------------------------------------------


class TestLadderConvergence:
    """Step convergence Spearman between identical maps should be 1.0."""

    def test_identical_maps_perfect_convergence(self):
        from igv.metrics import step_convergence_spearman

        L = 20
        rng = np.random.default_rng(123)
        pair_map = rng.normal(0, 1, (L, L)).astype(np.float64)
        pair_map = (pair_map + pair_map.T) / 2

        iu = np.triu_indices(L, k=1)
        flat = pair_map[iu]
        rho = step_convergence_spearman(flat, flat)
        assert rho == pytest.approx(1.0)

    def test_random_maps_low_convergence(self):
        from igv.metrics import step_convergence_spearman

        L = 50
        rng = np.random.default_rng(99)
        map1 = rng.normal(0, 1, (L, L))
        map2 = rng.normal(0, 1, (L, L))
        iu = np.triu_indices(L, k=1)
        rho = step_convergence_spearman(map1[iu], map2[iu])
        assert abs(rho) < 0.3


# ---------------------------------------------------------------------------
# 8. Dry run does not import boltz
# ---------------------------------------------------------------------------


class TestDryRun:
    """--dry-run must work without boltz installed."""

    def test_dry_run_returns_zero(self):
        ret = pair_capture.main(["--dry-run"])
        assert ret == 0

    def test_dry_run_custom_dataset(self):
        ret = pair_capture.main(["--dry-run", "--dataset", "1JTG"])
        assert ret == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
