#!/usr/bin/env python3
"""Merge Ridgey/Potts/ESMFold2-Fast results and package 48 NGLY1 designs."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
import zipfile
from pathlib import Path

import gemmi
import numpy as np


ROOT = Path("/home/ubuntu/codex_ngly1_20260819")
FOLD_DIR = ROOT / "work/esmfold2fast_fold64"
OUT_DIR = ROOT / "outputs/ngly1_48_10mut_designs"
N_FINAL = 48


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    name: str | None = None
    pieces: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                result[name] = "".join(pieces)
            name = line[1:].split()[0]
            pieces = []
        else:
            pieces.append(line.strip())
    if name is not None:
        result[name] = "".join(pieces)
    return result


def ca_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    structure = gemmi.read_structure(str(path))
    chain = next(iter(structure[0]))
    coordinates: list[list[float]] = []
    b_factors: list[float] = []
    for residue in chain:
        ca = residue.find_atom("CA", "*")
        if ca is None:
            coordinates.append([np.nan, np.nan, np.nan])
            b_factors.append(np.nan)
        else:
            coordinates.append([ca.pos.x, ca.pos.y, ca.pos.z])
            b_factors.append(float(ca.b_iso))
    return np.asarray(coordinates, dtype=np.float64), np.asarray(b_factors, dtype=np.float64)


def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask & np.isfinite(reference).all(axis=1) & np.isfinite(mobile).all(axis=1)
    ref = reference[valid]
    mob = mobile[valid]
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    ref0 = ref - ref_center
    mob0 = mob - mob_center
    u, _, vt = np.linalg.svd(mob0.T @ ref0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = mob0 @ rotation
    distances = np.linalg.norm(aligned - ref0, axis=1)
    return float(np.sqrt(np.mean(distances ** 2))), float(np.mean(distances <= 3.0))


def numeric(values: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in values]


def main() -> None:
    sequences = read_fasta(ROOT / "work/ngly1_fold64_prefold.fasta")
    prefold = {row["name"]: row for row in csv.DictReader((ROOT / "work/ngly1_fold64_prefold.csv").open())}
    folds: dict[str, dict] = {}
    for path in sorted(FOLD_DIR.glob("shard_*.json")):
        for row in json.loads(path.read_text()):
            folds[row["name"]] = row
    if len(folds) != len(sequences):
        raise RuntimeError(f"expected {len(sequences)} fold records, found {len(folds)}")

    wt_fold = folds["WT"]
    wt_coordinates, _ = ca_coordinates(Path(wt_fold["cif"]))
    afdb_coordinates, afdb_plddt = ca_coordinates(ROOT / "ngly1_afdb.cif")
    if len(afdb_coordinates) != len(wt_coordinates):
        raise ValueError("AFDB and ESMFold2-Fast structures have different lengths")
    positions = np.arange(1, len(wt_coordinates) + 1)
    high_confidence = np.isfinite(afdb_plddt) & (afdb_plddt >= 85.0) & ~((positions >= 112) & (positions <= 168))
    domains = {
        "nterm": high_confidence & (positions <= 111),
        "core": high_confidence & (positions >= 169) & (positions <= 453),
        "paw": high_confidence & (positions >= 454),
    }

    evaluated: list[dict] = []
    for source_name, row in prefold.items():
        fold = folds[source_name]
        coordinates, _ = ca_coordinates(Path(fold["cif"]))
        global_rmsd, global_fraction = kabsch_rmsd(wt_coordinates, coordinates, high_confidence)
        domain_rmsds = {name: kabsch_rmsd(wt_coordinates, coordinates, mask)[0] for name, mask in domains.items()}
        plddt_delta = float(fold["plddt"]) - float(wt_fold["plddt"])
        ptm_delta = float(fold["ptm"]) - float(wt_fold["ptm"])
        fold_score = (
            float(row["selection_score"])
            + 10.0 * plddt_delta
            + 5.0 * ptm_delta
            - 0.08 * max(domain_rmsds.values())
            - 0.02 * global_rmsd
        )
        evaluated.append({
            **row,
            "source_name": source_name,
            "sequence": sequences[source_name],
            "esmfold2fast_plddt": float(fold["plddt"]),
            "esmfold2fast_plddt_delta_vs_wt": plddt_delta,
            "esmfold2fast_ptm": float(fold["ptm"]),
            "esmfold2fast_ptm_delta_vs_wt": ptm_delta,
            "structured_ca_rmsd_vs_wt_angstrom": global_rmsd,
            "structured_ca_fraction_within_3A": global_fraction,
            "nterm_ca_rmsd_angstrom": domain_rmsds["nterm"],
            "core_ca_rmsd_angstrom": domain_rmsds["core"],
            "paw_ca_rmsd_angstrom": domain_rmsds["paw"],
            "max_domain_ca_rmsd_angstrom": max(domain_rmsds.values()),
            "fold_selection_score": fold_score,
            "source_cif": fold["cif"],
        })

    passing = [
        row for row in evaluated
        if row["esmfold2fast_plddt"] >= float(wt_fold["plddt"]) - 0.01
        and row["esmfold2fast_ptm"] >= float(wt_fold["ptm"]) - 0.03
        and row["max_domain_ca_rmsd_angstrom"] <= 2.5
    ]
    selected = sorted(passing, key=lambda row: row["fold_selection_score"], reverse=True)[:N_FINAL]
    if len(selected) < N_FINAL:
        raise RuntimeError(f"only {len(selected)} designs passed folding gates")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    structures_dir = OUT_DIR / "structures"
    structures_dir.mkdir(parents=True)
    shutil.copy2(wt_fold["cif"], structures_dir / "WT_esmfold2fast.cif")

    fasta_path = OUT_DIR / "ngly1_48_10mut_designs.fasta"
    csv_path = OUT_DIR / "ngly1_48_10mut_designs.csv"
    manifest_path = OUT_DIR / "ngly1_48_manifest.json"
    report_path = OUT_DIR / "README.md"
    final_rows: list[dict] = []
    with fasta_path.open("w") as fasta_handle:
        for rank, row in enumerate(selected, 1):
            final_name = f"NGLY1_10M_{rank:03d}"
            mutations = row["mutations"].split(";")
            if len(mutations) != 10:
                raise ValueError(f"{row['source_name']} does not have ten mutations")
            fasta_handle.write(
                f">{final_name} source={row['source_name']} mutations={'|'.join(mutations)}\n{row['sequence']}\n"
            )
            destination = structures_dir / f"{final_name}_esmfold2fast.cif"
            shutil.copy2(row["source_cif"], destination)
            final_rows.append({
                "design": final_name,
                **{key: value for key, value in row.items() if key not in {"sequence", "source_cif", "name"}},
                "structure_file": f"structures/{destination.name}",
            })

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    manifest_path.write_text(json.dumps({
        "target": "human NGLY1 / UniProt Q96IV0",
        "wild_type_length": len(sequences["WT"]),
        "design_count": len(final_rows),
        "mutations_per_design": 10,
        "protected_catalytic_triad": [309, 336, 353],
        "protected_zinc_ligands": [250, 253, 283, 286],
        "catalytic_shell_ca_cutoff_angstrom": 18.0,
        "ridgey_models": ["600m", "600m_ens1", "600m_ens2", "600m_ens3", "600m_ens4"],
        "potts": {"msa_sequences": 5000, "rank": 32, "selected_l2_lambda": 1.0},
        "esmfold2fast_wt": {"plddt": wt_fold["plddt"], "ptm": wt_fold["ptm"]},
        "selection": {
            "stability": "positive delta in all five Ridgey 600M replicas",
            "ec": "positive ensemble-mean P(EC 3.5.1.52), non-decreasing in >=4/5 replicas, worst delta >= -0.0018",
            "solubility": "positive ensemble-mean delta",
            "lm": "positive ensemble-mean mutation-site masked log-probability delta",
            "fold": "ESMFold2-Fast pLDDT >= WT-0.01, pTM >= WT-0.03, max per-domain CA RMSD <=2.5 A",
        },
        "files": {"fasta": fasta_path.name, "csv": csv_path.name, "structures": "structures/"},
    }, indent=2) + "\n")

    st_mean = numeric(final_rows, "ridgey_stability_delta_mean")
    st_min = numeric(final_rows, "ridgey_stability_delta_min")
    ec_mean = numeric(final_rows, "ridgey_ec_delta_mean")
    sol_mean = numeric(final_rows, "ridgey_solubility_delta_mean")
    plddt = numeric(final_rows, "esmfold2fast_plddt")
    ptm = numeric(final_rows, "esmfold2fast_ptm")
    core_rmsd = numeric(final_rows, "core_ca_rmsd_angstrom")
    report_path.write_text(f"""# NGLY1 48 × ten-mutation design set

This directory contains 48 exact ten-substitution variants of human NGLY1 (Q96IV0; 654 aa). These are computational candidates, not experimentally validated proteins.

## Hard constraints used

- No substitution occurs in the catalytic triad (309, 336, 353), the Zn ligands (250, 253, 283, 286), the entire 18 Å CA shell around those sites, or Ridgey contact/PTM protection masks.
- Stability improves against WT in every one of five local Ridgey v2 600M replicas.
- Mean P(EC 3.5.1.52) improves; at least four of five replicas are non-decreasing, with the single-replica tolerance capped at 0.0018.
- Mean solubility and sequence-only mutation-site masked log-probability improve.
- Every substitution is observed in the 5,000-sequence MSA and complete designs were ranked by the rank-32, L2-regularized Potts model (selected λ=1.0).
- ESMFold2-Fast predicts WT-like folding using three recycling loops and 50 sampling steps.

## Final-set ranges

- Ridgey stability Δ vs WT, ensemble mean: {min(st_mean):.3f} to {max(st_mean):.3f} kcal/mol-equivalent scale
- Worst-replica Ridgey stability Δ: {min(st_min):.3f} to {max(st_min):.3f}
- Mean P(EC 3.5.1.52) Δ: {min(ec_mean):.6f} to {max(ec_mean):.6f}
- Mean solubility Δ: {min(sol_mean):.4f} to {max(sol_mean):.4f}
- ESMFold2-Fast pLDDT: {min(plddt):.4f} to {max(plddt):.4f} (WT {float(wt_fold['plddt']):.4f})
- ESMFold2-Fast pTM: {min(ptm):.4f} to {max(ptm):.4f} (WT {float(wt_fold['ptm']):.4f})
- Catalytic-core CA RMSD to the WT prediction: {min(core_rmsd):.3f} to {max(core_rmsd):.3f} Å

Use `ngly1_48_10mut_designs.csv` for per-design audit columns and `structures/` for the predicted mmCIF files. Experimental expression, activity, metal loading, and thermal-shift testing are still required before calling any design stabilized in the laboratory.
""")

    archive_path = OUT_DIR / "ngly1_48_structures.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(structures_dir.glob("*.cif")):
            archive.write(path, arcname=path.name)
    print(json.dumps({
        "folded": len(evaluated),
        "fold_passing": len(passing),
        "selected": len(final_rows),
        "output_dir": str(OUT_DIR),
        "wt_plddt": wt_fold["plddt"],
        "wt_ptm": wt_fold["ptm"],
    }, indent=2))


if __name__ == "__main__":
    main()
