#!/usr/bin/env python3
"""Predict mutation effects using gradient attribution and embedding deltas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.attrib import predict_mutants, score_deltas
from igv.provenance import read as prov_read, write as prov_write


def _parse_substitutions(s: str) -> tuple[tuple[int, str], ...]:
    if not s or pd.isna(s):
        return ()
    parts = s.split(";")
    return tuple((int(p[:-1]), p[-1]) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict mutant binding scores from gradient attribution"
    )
    parser.add_argument("--library", required=True, help="Library parquet from 01_build_library")
    parser.add_argument("--grad", required=True, help="Gradient .npz from 03_attribute")
    parser.add_argument("--deltas", required=True, help="Embedding deltas .npz from 02_embed_deltas")
    parser.add_argument(
        "--out",
        default="results/pred.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    library_path = Path(args.library)
    grad_path = Path(args.grad)
    deltas_path = Path(args.deltas)
    out_path = Path(args.out)

    for p, label in [(library_path, "library"), (grad_path, "grad"), (deltas_path, "deltas")]:
        if not p.exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    grad_prov = prov_read(grad_path)
    arm = grad_prov.get("arm", {})

    print(f"Loading library from {library_path}...")
    df = pd.read_parquet(library_path)
    substitutions = [_parse_substitutions(s) for s in df["substitutions_str"]]

    print(f"Loading gradient from {grad_path}...")
    grad_data = np.load(grad_path)
    grad_key = "grad" if "grad" in grad_data else list(grad_data.keys())[0]
    grad = grad_data[grad_key]

    print(f"Loading deltas from {deltas_path}...")
    deltas_data = np.load(deltas_path, allow_pickle=True)

    delta_emb: dict[tuple[int, str], np.ndarray] = {}
    if "keys" in deltas_data and "values" in deltas_data:
        keys = deltas_data["keys"]
        values = deltas_data["values"]
        for k, v in zip(keys, values):
            pos, aa = int(k[0]), str(k[1])
            delta_emb[(pos, aa)] = v
    elif "delta_emb" in deltas_data:
        raw = deltas_data["delta_emb"].item()
        delta_emb = {(int(k[0]), str(k[1])): v for k, v in raw.items()}
    else:
        for key in deltas_data.files:
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                pos, aa = int(parts[0]), parts[1]
                delta_emb[(pos, aa)] = deltas_data[key]

    print(f"  {len(delta_emb)} per-substitution deltas loaded")
    print(f"  gradient shape: {grad.shape}")

    if grad.ndim == 3:
        grad = grad[0]

    print("Computing score deltas...")
    sd = score_deltas(grad, delta_emb)
    print(f"  {len(sd)} score deltas computed")

    print("Predicting mutants...")
    preds = predict_mutants(sd, substitutions)

    out_df = pd.DataFrame({
        "pred": preds,
        "binding_score": df["binding_score"].values,
        "n_mut": df["n_mut"].values,
        "substitutions": df["substitutions_str"].values,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_df)} rows, {out_path.stat().st_size / 1024:.1f} KB)")

    prov_write(
        out_path,
        stage="05_predict",
        inputs={
            "library": str(library_path),
            "grad": str(grad_path),
            "deltas": str(deltas_path),
        },
        params={},
        arm=arm,
    )


if __name__ == "__main__":
    main()
