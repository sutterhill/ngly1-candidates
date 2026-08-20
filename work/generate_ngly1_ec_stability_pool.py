#!/usr/bin/env python3
"""Recombine locally scored singles into exact ten-mutation NGLY1 designs."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser


ROOT = Path("/home/ubuntu/codex_ngly1_20260819")
SEED = 20260820
POOL_SIZE = 240
AA = "ACDEFGHIKLMNPQRSTVWY"
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}
HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2,
    "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}


def load_ensemble(directory: Path) -> tuple[list[dict], dict[str, dict]]:
    models = [json.loads((directory / f"{name}.json").read_text()) for name in ("base", "ens1", "ens2", "ens3", "ens4")]
    wild_types = [model["records"][0] for model in models]
    names = [record["name"] for record in models[0]["records"][1:]]
    scored: dict[str, dict] = {}
    for name in names:
        records = [next(record for record in model["records"] if record["name"] == name) for model in models]
        stability = [record["stability"] - wt["stability"] for record, wt in zip(records, wild_types)]
        ec = [record["ec_3.5.1.52"] - wt["ec_3.5.1.52"] for record, wt in zip(records, wild_types)]
        solubility = [record["solubility"] - wt["solubility"] for record, wt in zip(records, wild_types)]
        lm = [record["masked_lm"]["delta_mean_logp_vs_wt"] for record in records]
        scored[name] = {
            "stability": stability,
            "ec": ec,
            "solubility": solubility,
            "lm": lm,
            "active_sites_intact": all(record["active_sites_1_indexed"] == wt["active_sites_1_indexed"] for record, wt in zip(records, wild_types)),
        }
    return wild_types, scored


def main() -> None:
    rng = random.Random(SEED)
    sequence = "".join(line.strip() for line in (ROOT / "ngly1.fasta").read_text().splitlines() if not line.startswith(">"))
    _, scored = load_ensemble(ROOT / "work/ridgey_local_ensemble_broad_singles")
    metadata = {row["mutation"]: row for row in csv.DictReader((ROOT / "work/ngly1_broad_single_pool.csv").open())}

    ev = np.load(ROOT / "ngly1_ev_h100.npz")
    ev_aa = str(ev["aa_order"])
    h, v = ev["h"], ev["V"]
    query_states = np.asarray([1 + ev_aa.index(amino_acid) for amino_acid in sequence], dtype=np.int64)
    baseline_v = v[np.arange(len(sequence)), :, query_states]
    baseline_total = baseline_v.sum(axis=0)

    structure = MMCIFParser(QUIET=True).get_structure("ngly1", str(ROOT / "ngly1_afdb.cif"))
    residues = [residue for residue in next(structure.get_chains()) if residue.id[0] == " "]
    ca = np.asarray([residue["CA"].coord for residue in residues])

    candidates: list[dict] = []
    for mutation, values in scored.items():
        meta = metadata[mutation]
        mean_ec = statistics.mean(values["ec"])
        mean_stability = statistics.mean(values["stability"])
        mean_lm = statistics.mean(values["lm"])
        if not values["active_sites_intact"]:
            continue
        if mean_ec < 0.0 or min(values["ec"]) < -0.0017:
            continue
        if mean_stability < -0.02 or min(values["stability"]) < -0.06:
            continue
        if mean_lm < -1.0:
            continue
        position = int(meta["position"])
        native, mutant = meta["native"], meta["mutant"]
        old_state = query_states[position - 1]
        new_state = 1 + ev_aa.index(mutant)
        old_v = v[position - 1, :, old_state]
        new_v = v[position - 1, :, new_state]
        delta_v = new_v - old_v
        potts_field_delta = float(h[position - 1, new_state] - h[position - 1, old_state])
        potts_norm_delta = float(np.dot(new_v, new_v) - np.dot(old_v, old_v))
        surface_hydropathy_delta = (
            HYDROPATHY[mutant] - HYDROPATHY[native]
            if float(meta["relative_sasa"]) > 0.35 else 0.0
        )
        mutation_score = (
            500.0 * mean_ec
            + 3.0 * mean_stability
            + 0.10 * mean_lm
            + 2.0 * statistics.mean(values["solubility"])
            + 0.02 * float(meta["potts_single_delta"])
            + 0.08 * math.log10(1.0 + 5000.0 * float(meta["msa_frequency"]))
        )
        candidates.append({
            "mutation": mutation,
            "position": position,
            "native": native,
            "mutant": mutant,
            "stability": values["stability"],
            "ec": values["ec"],
            "solubility": values["solubility"],
            "lm": values["lm"],
            "mean_stability": mean_stability,
            "mean_ec": mean_ec,
            "mean_solubility": statistics.mean(values["solubility"]),
            "mean_lm": mean_lm,
            "msa_frequency": float(meta["msa_frequency"]),
            "independent_delta": float(meta["independent_delta"]),
            "potts_single_delta": float(meta["potts_single_delta"]),
            "delta_v": delta_v,
            "potts_field_delta": potts_field_delta,
            "potts_norm_delta": potts_norm_delta,
            "charge_delta": CHARGE.get(mutant, 0) - CHARGE.get(native, 0),
            "surface_hydropathy_delta": surface_hydropathy_delta,
            "score": mutation_score,
        })

    by_position: dict[int, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_position[candidate["position"]].append(candidate)
    positions = sorted(by_position)
    position_weights = [math.exp(min(3.0, max(-3.0, max(row["score"] for row in by_position[position])))) for position in positions]

    designs: dict[tuple[str, ...], dict] = {}
    for _ in range(250_000):
        remaining_positions = positions.copy()
        remaining_weights = position_weights.copy()
        chosen_positions: list[int] = []
        for _ in range(10):
            position = rng.choices(remaining_positions, weights=remaining_weights, k=1)[0]
            offset = remaining_positions.index(position)
            chosen_positions.append(position)
            remaining_positions.pop(offset)
            remaining_weights.pop(offset)
        chosen: list[dict] = []
        for position in chosen_positions:
            options = by_position[position]
            weights = [math.exp(min(3.0, max(-3.0, row["score"]))) for row in options]
            chosen.append(rng.choices(options, weights=weights, k=1)[0])

        if sum(row["position"] < 454 for row in chosen) < 3:
            continue
        if any(
            np.linalg.norm(ca[left["position"] - 1] - ca[right["position"] - 1]) < 6.0
            for index, left in enumerate(chosen)
            for right in chosen[index + 1 :]
        ):
            continue
        net_charge = sum(row["charge_delta"] for row in chosen)
        if abs(net_charge) > 3:
            continue
        surface_hydropathy = sum(row["surface_hydropathy_delta"] for row in chosen)
        if surface_hydropathy > 1.5:
            continue

        additive_stability = [sum(row["stability"][model] for row in chosen) for model in range(5)]
        additive_ec = [sum(row["ec"][model] for row in chosen) for model in range(5)]
        additive_solubility = [sum(row["solubility"][model] for row in chosen) for model in range(5)]
        additive_lm = [sum(row["lm"][model] for row in chosen) for model in range(5)]
        if min(additive_stability) < 0.0:
            continue
        if statistics.mean(additive_ec) < 0.0015 or min(additive_ec) < -0.012:
            continue
        if statistics.mean(additive_solubility) < -0.04:
            continue
        if statistics.mean(additive_lm) < 0.0:
            continue

        total_delta_v = np.sum([row["delta_v"] for row in chosen], axis=0)
        potts_delta = (
            sum(row["potts_field_delta"] for row in chosen)
            + float(np.dot(baseline_total, total_delta_v))
            + 0.5 * float(np.dot(total_delta_v, total_delta_v))
            - 0.5 * sum(row["potts_norm_delta"] for row in chosen)
        )
        independent_delta = sum(row["independent_delta"] for row in chosen)
        mutations = tuple(sorted((row["mutation"] for row in chosen), key=lambda text: int(text[1:-1])))
        score = (
            350.0 * statistics.mean(additive_ec)
            + 1.8 * min(additive_stability)
            + 0.8 * statistics.mean(additive_stability)
            + 1.5 * statistics.mean(additive_solubility)
            + 0.03 * statistics.mean(additive_lm)
            + 0.025 * potts_delta
            + 0.01 * independent_delta
            - 0.03 * abs(net_charge)
        )
        current = designs.get(mutations)
        if current is None or score > current["generation_score"]:
            designs[mutations] = {
                "mutations": mutations,
                "chosen": chosen,
                "generation_score": score,
                "additive_ec_mean": statistics.mean(additive_ec),
                "additive_ec_min": min(additive_ec),
                "additive_stability_mean": statistics.mean(additive_stability),
                "additive_stability_min": min(additive_stability),
                "additive_solubility_mean": statistics.mean(additive_solubility),
                "additive_lm_mean": statistics.mean(additive_lm),
                "potts_delta": potts_delta,
                "independent_delta": independent_delta,
                "net_charge_delta": net_charge,
                "surface_hydropathy_delta": surface_hydropathy,
            }

    ranked = sorted(designs.values(), key=lambda row: row["generation_score"], reverse=True)
    selected: list[dict] = []
    mutation_use: Counter[str] = Counter()
    for design in ranked:
        mutation_set = set(design["mutations"])
        if any(len(mutation_set & set(other["mutations"])) > 8 for other in selected):
            continue
        if any(mutation_use[mutation] >= int(0.95 * POOL_SIZE) for mutation in mutation_set):
            continue
        selected.append(design)
        mutation_use.update(mutation_set)
        if len(selected) == POOL_SIZE:
            break
    if len(selected) < POOL_SIZE:
        raise RuntimeError(f"selected only {len(selected)} designs from {len(designs)} generated")

    fieldnames = [
        "name", "mutations", "generation_score", "additive_ec_mean", "additive_ec_min",
        "additive_stability_mean", "additive_stability_min", "additive_solubility_mean",
        "additive_lm_mean", "potts_delta", "independent_delta", "net_charge_delta",
        "surface_hydropathy_delta", "minimum_catalytic_ca_distance",
    ]
    csv_path = ROOT / "work/ngly1_ec_stability_10mut_pool.csv"
    fasta_path = ROOT / "work/ngly1_ec_stability_10mut_pool.fasta"
    catalytic_and_zinc = [250, 253, 283, 286, 309, 336, 353]
    with csv_path.open("w", newline="") as csv_handle, fasta_path.open("w") as fasta_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, design in enumerate(selected, 1):
            name = f"NGLY1_EC{index:03d}"
            designed = list(sequence)
            for row in design["chosen"]:
                designed[row["position"] - 1] = row["mutant"]
            minimum_catalytic_distance = min(
                np.linalg.norm(ca[row["position"] - 1] - ca[site - 1])
                for row in design["chosen"] for site in catalytic_and_zinc
            )
            writer.writerow({
                "name": name,
                "mutations": ";".join(design["mutations"]),
                "generation_score": design["generation_score"],
                "additive_ec_mean": design["additive_ec_mean"],
                "additive_ec_min": design["additive_ec_min"],
                "additive_stability_mean": design["additive_stability_mean"],
                "additive_stability_min": design["additive_stability_min"],
                "additive_solubility_mean": design["additive_solubility_mean"],
                "additive_lm_mean": design["additive_lm_mean"],
                "potts_delta": design["potts_delta"],
                "independent_delta": design["independent_delta"],
                "net_charge_delta": design["net_charge_delta"],
                "surface_hydropathy_delta": design["surface_hydropathy_delta"],
                "minimum_catalytic_ca_distance": minimum_catalytic_distance,
            })
            fasta_handle.write(f">{name} mutations={'|'.join(design['mutations'])}\n{''.join(designed)}\n")

    print(json.dumps({
        "eligible_mutations": len(candidates),
        "eligible_positions": len(positions),
        "unique_designs": len(designs),
        "selected": len(selected),
        "top_score": selected[0]["generation_score"],
        "last_score": selected[-1]["generation_score"],
        "csv": str(csv_path),
        "fasta": str(fasta_path),
    }, indent=2))


if __name__ == "__main__":
    main()
