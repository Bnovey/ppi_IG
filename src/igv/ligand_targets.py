"""Target registry for the protein-ligand affinity IG arm.

Lists ABL1, EGFR (erlotinib + gefitinib), PimA (GDD + GDP), and
Mpro (nirmatrelvir) with all metadata needed by the gate/scan scripts:
PDB, chain(s), ligand CCD, residue-number offset, sign convention,
ground truth path, and per-position distance to ligand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LigandTarget:
    name: str
    protein: str
    organism: str
    pdb_id: str
    chains: tuple[str, ...]
    ligand_ccd: str
    ligand_chain: str
    resnum_offset: int
    construct_span: tuple[int, int]
    sign_convention: str
    assay: str
    ground_truth_path: str
    source: str
    log_base: str
    notes: str = ""
    position_distances: dict[int, float] = field(default_factory=dict)


TARGETS: dict[str, LigandTarget] = {
    "ABL1-imatinib": LigandTarget(
        name="ABL1-imatinib",
        protein="ABL1 kinase domain",
        organism="H. sapiens",
        pdb_id="2HYY",
        chains=("A",),
        ligand_ccd="STI",
        ligand_chain="A",
        resnum_offset=0,
        construct_span=(235, 498),
        sign_convention="positive=resistance",
        assay="cell-viability IC50 (Ba/F3), DDG at 300K",
        ground_truth_path="data/raw/hauser2018_abl_supp3.csv",
        source="Hauser et al. 2018, doi:10.1038/s42003-018-0075-x",
        log_base="log10_via_RT",
    ),
    "EGFR-erlotinib": LigandTarget(
        name="EGFR-erlotinib",
        protein="EGFR kinase domain",
        organism="H. sapiens",
        pdb_id="1M17",
        chains=("A",),
        ligand_ccd="AQ4",
        ligand_chain="A",
        resnum_offset=-24,
        construct_span=(672, 995),
        sign_convention="positive=resistance",
        assay="cell-viability IC50 ratio (Ba/F3), 3 replicates",
        ground_truth_path="data/processed/egfr_ground_truth.csv",
        source="Robichaux et al. 2021, doi:10.1038/s41586-021-03898-1",
        log_base="unknown",
        notes="1M17 numbering offset -24 from clinical/precursor frame",
        position_distances={
            709: 14.72, 718: 3.27, 719: 4.50, 720: 6.55, 724: 9.44,
            725: 7.63, 747: 10.13, 754: 15.31, 757: 13.02, 761: 10.87,
            768: 12.17, 769: 11.06, 774: 10.21, 776: 6.68, 784: 15.64,
            790: 3.41, 811: 21.32, 833: 12.90, 858: 10.12, 861: 15.68,
        },
    ),
    "EGFR-gefitinib": LigandTarget(
        name="EGFR-gefitinib",
        protein="EGFR kinase domain",
        organism="H. sapiens",
        pdb_id="1M17",
        chains=("A",),
        ligand_ccd="AQ4",
        ligand_chain="A",
        resnum_offset=-24,
        construct_span=(672, 995),
        sign_convention="positive=resistance",
        assay="cell-viability IC50 ratio (Ba/F3), 3 replicates",
        ground_truth_path="data/processed/egfr_ground_truth.csv",
        source="Robichaux et al. 2021, doi:10.1038/s41586-021-03898-1",
        log_base="unknown",
        notes="Same structure as erlotinib (1M17/AQ4); gefitinib data from same XLSX",
        position_distances={
            709: 14.72, 718: 3.27, 719: 4.50, 720: 6.55, 724: 9.44,
            725: 7.63, 747: 10.13, 754: 15.31, 757: 13.02, 761: 10.87,
            768: 12.17, 769: 11.06, 774: 10.21, 776: 6.68, 784: 15.64,
            790: 3.41, 811: 21.32, 833: 12.90, 858: 10.12, 861: 15.68,
        },
    ),
    "PimA-GDD": LigandTarget(
        name="PimA-GDD",
        protein="PimA mannosyltransferase",
        organism="C. glutamicum",
        pdb_id="3OKP",
        chains=("A",),
        ligand_ccd="GDD",
        ligand_chain="A",
        resnum_offset=0,
        construct_span=(4, 381),
        sign_convention="positive=resistance",
        assay="Kd, Trp fluorescence",
        ground_truth_path="data/processed/pima_gdd_ground_truth.csv",
        source="Batt et al. 2010, doi:10.1074/jbc.M110.165407, via Platinum",
        log_base="log10",
        position_distances={
            12: 16.66, 13: 13.42, 20: 6.50, 21: 8.48, 22: 10.63,
            118: 9.47, 123: 14.55, 205: 3.87, 206: 2.74, 210: 3.34,
            211: 2.77, 290: 3.64, 291: 6.53,
        },
    ),
    "PimA-GDP": LigandTarget(
        name="PimA-GDP",
        protein="PimA mannosyltransferase",
        organism="C. glutamicum",
        pdb_id="3OKC",
        chains=("A",),
        ligand_ccd="GDP",
        ligand_chain="A",
        resnum_offset=0,
        construct_span=(4, 381),
        sign_convention="positive=resistance",
        assay="Kd, Trp fluorescence",
        ground_truth_path="data/processed/pima_gdp_ground_truth.csv",
        source="Batt et al. 2010, doi:10.1074/jbc.M110.165407, via Platinum",
        log_base="log10",
        position_distances={
            12: 16.85, 13: 13.40, 20: 6.55, 21: 8.34, 22: 10.83,
            118: 8.82, 123: 14.18, 205: 3.98, 206: 3.45, 210: 3.65,
            211: 2.91, 290: 4.16, 291: 6.80,
        },
    ),
    "Mpro-nirmatrelvir": LigandTarget(
        name="Mpro-nirmatrelvir",
        protein="Main protease (Mpro/3CLpro)",
        organism="SARS-CoV-2",
        pdb_id="8DZ2",
        chains=("A", "B"),
        ligand_ccd="4WI",
        ligand_chain="A",
        resnum_offset=0,
        construct_span=(1, 302),
        sign_convention="positive=resistance",
        assay="FRET enzymatic Ki",
        ground_truth_path="data/raw/mpro_ki.csv",
        source="Hu et al. 2023, doi:10.1021/acscentsci.3c00538",
        log_base="log10",
        notes="Dimer: E166 and H172 contact partner protomer. Use 8DZ2, not 7VH8.",
    ),
}


def get_target(name: str) -> LigandTarget:
    if name not in TARGETS:
        raise KeyError(
            f"Unknown target {name!r}. Available: {sorted(TARGETS)}"
        )
    return TARGETS[name]


def compute_ligand_distances(
    pdb_path: Path,
    protein_chain: str,
    ligand_ccd: str,
    positions: list[int],
    ligand_chain: str | None = None,
) -> dict[int, float]:
    """Compute minimum heavy-atom distance from each position to the ligand.

    Parses ATOM records for the protein chain and HETATM records for the
    ligand (identified by CCD code). Returns {resnum: min_distance_angstrom}.
    """
    lig_coords: list[tuple[float, float, float]] = []
    res_coords: dict[int, list[tuple[float, float, float]]] = {}

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                chain = line[21]
                if resname == ligand_ccd:
                    if ligand_chain is not None and chain != ligand_chain:
                        continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        lig_coords.append((x, y, z))
                    except ValueError:
                        pass

            elif line.startswith("ATOM"):
                chain = line[21]
                if chain != protein_chain:
                    continue
                element = line[76:78].strip() if len(line) > 76 else ""
                if element == "H":
                    continue
                resnum_str = line[22:26].strip()
                try:
                    resnum = int(resnum_str)
                except ValueError:
                    continue
                if resnum not in positions:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_coords.setdefault(resnum, []).append((x, y, z))
                except ValueError:
                    pass

    if not lig_coords:
        raise ValueError(
            f"No HETATM records for ligand {ligand_ccd} "
            f"(chain={ligand_chain}) in {pdb_path}"
        )

    lig_arr = np.array(lig_coords)

    distances: dict[int, float] = {}
    for resnum in positions:
        if resnum not in res_coords:
            distances[resnum] = float("nan")
            continue
        res_arr = np.array(res_coords[resnum])
        dists = np.sqrt(
            ((res_arr[:, None, :] - lig_arr[None, :, :]) ** 2).sum(axis=2)
        )
        distances[resnum] = float(dists.min())

    return distances
