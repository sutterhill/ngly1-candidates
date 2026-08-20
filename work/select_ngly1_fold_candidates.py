#!/usr/bin/env python3
"""Select diverse consensus Ridgey winners for local structure prediction."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("/home/ubuntu/codex_ngly1_20260819")
MODEL_NAMES = ("base", "ens1", "ens2", "ens3", "ens4")
N_FOLD = 64


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


def joined(values: list[float]) -> str:
    return ";".join(f"{value:.9g}" for value in values)


def main() -> None:
    directory = ROOT / "work/ridgey_local_ensemble_ec_pool10"
    models = [json.loads((directory / f"{name}.json").read_text()) for name in MODEL_NAMES]
    wild_types = [model["records"][0] for model in models]
    record_maps = [{record["name"]: record for record in model["records"]} for model in models]
    sequences = read_fasta(ROOT / "work/ngly1_ec_stability_10mut_pool.fasta")
    metadata = {row["name"]: row for row in csv.DictReader((ROOT / "work/ngly1_ec_stability_10mut_pool.csv").open())}

    passing: list[dict] = []
    for name, sequence in sequences.items():
        records = [mapping[name] for mapping in record_maps]
        stability = [record["stability"] - wt["stability"] for record, wt in zip(records, wild_types)]
        ec_absolute = [record["ec_3.5.1.52"] for record in records]
        ec = [value - wt["ec_3.5.1.52"] for value, wt in zip(ec_absolute, wild_types)]
        solubility = [record["solubility"] - wt["solubility"] for record, wt in zip(records, wild_types)]
        lm = [record["masked_lm"]["delta_mean_logp_vs_wt"] for record in records]
        ppl_ratio = [record["masked_lm"]["ppl_ratio_vs_wt"] for record in records]
        enzyme = [record["is_enzyme"] for record in records]
        zinc = [record["zinc"] for record in records]
        active_intact = all(record["active_sites_1_indexed"] == wt["active_sites_1_indexed"] for record, wt in zip(records, wild_types))
        ec_up_models = sum(value >= 0.0 for value in ec)
        if min(stability) <= 0.0:
            continue
        if statistics.mean(ec) < 0.0 or min(ec) < -0.0018 or ec_up_models < 4:
            continue
        if statistics.mean(solubility) < 0.0:
            continue
        if statistics.mean(lm) < 0.0:
            continue
        if min(enzyme) < 0.99 or min(zinc) < 0.50 or not active_intact:
            continue
        meta = metadata[name]
        score = (
            500.0 * statistics.mean(ec)
            + 2.0 * min(stability)
            + 0.8 * statistics.mean(stability)
            + 5.0 * statistics.mean(solubility)
            + 0.05 * statistics.mean(lm)
            + 0.01 * float(meta["potts_delta"])
        )
        passing.append({
            "name": name,
            "sequence": sequence,
            "mutations": meta["mutations"],
            "selection_score": score,
            "ridgey_stability_delta_mean": statistics.mean(stability),
            "ridgey_stability_delta_min": min(stability),
            "ridgey_stability_delta_by_model": joined(stability),
            "ridgey_ec_probability_mean": statistics.mean(ec_absolute),
            "ridgey_ec_delta_mean": statistics.mean(ec),
            "ridgey_ec_delta_min": min(ec),
            "ridgey_ec_non_decreasing_models": ec_up_models,
            "ridgey_ec_delta_by_model": joined(ec),
            "ridgey_solubility_delta_mean": statistics.mean(solubility),
            "ridgey_solubility_delta_min": min(solubility),
            "ridgey_solubility_delta_by_model": joined(solubility),
            "ridgey_masked_lm_delta_mean": statistics.mean(lm),
            "ridgey_masked_ppl_ratio_mean": statistics.mean(ppl_ratio),
            "ridgey_enzyme_probability_min": min(enzyme),
            "ridgey_zinc_probability_min": min(zinc),
            "active_sites_all_models": "309;336;353",
            "potts_delta": float(meta["potts_delta"]),
            "independent_delta": float(meta["independent_delta"]),
            "net_charge_delta": int(meta["net_charge_delta"]),
            "surface_hydropathy_delta": float(meta["surface_hydropathy_delta"]),
            "minimum_catalytic_ca_distance": float(meta["minimum_catalytic_ca_distance"]),
        })

    selected: list[dict] = []
    for row in sorted(passing, key=lambda item: item["selection_score"], reverse=True):
        mutations = set(row["mutations"].split(";"))
        if any(len(mutations & set(other["mutations"].split(";"))) > 8 for other in selected):
            continue
        selected.append(row)
        if len(selected) == N_FOLD:
            break
    if len(selected) < N_FOLD:
        raise RuntimeError(f"selected only {len(selected)} fold candidates from {len(passing)} passing")

    csv_path = ROOT / "work/ngly1_fold64_prefold.csv"
    fasta_path = ROOT / "work/ngly1_fold64_prefold.fasta"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [key for key in selected[0] if key != "sequence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: value for key, value in row.items() if key != "sequence"})
    wild_type_sequence = "".join(line.strip() for line in (ROOT / "ngly1.fasta").read_text().splitlines() if not line.startswith(">"))
    with fasta_path.open("w") as handle:
        handle.write(f">WT\n{wild_type_sequence}\n")
        for row in selected:
            handle.write(f">{row['name']} mutations={row['mutations'].replace(';', '|')}\n{row['sequence']}\n")
    print(json.dumps({
        "passing": len(passing),
        "selected_for_folding": len(selected),
        "csv": str(csv_path),
        "fasta": str(fasta_path),
    }, indent=2))


if __name__ == "__main__":
    main()
