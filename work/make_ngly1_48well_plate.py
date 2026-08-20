#!/usr/bin/env python3
"""Create a balanced 6x8 plate assignment for the 48 final NGLY1 designs."""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path("/home/ubuntu/codex_ngly1_20260819")
OUT = ROOT / "outputs/ngly1_48_10mut_designs"
ROWS = "ABCDEF"
COLS = range(1, 9)
SEED = 20260820


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    pieces: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(pieces)
            name = line[1:].split()[0]
            pieces = []
        else:
            pieces.append(line.strip())
    if name is not None:
        records[name] = "".join(pieces)
    return records


def percentile_scores(rows: list[dict], key: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row[key]))
    denominator = max(1, len(ordered) - 1)
    return {row["design"]: index / denominator for index, row in enumerate(ordered)}


def main() -> None:
    rows = list(csv.DictReader((OUT / "ngly1_48_10mut_designs.csv").open()))
    sequences = read_fasta(OUT / "ngly1_48_10mut_designs.fasta")
    if len(rows) != 48 or len(sequences) != 48:
        raise ValueError("the final set must contain exactly 48 designs")
    names = [row["design"] for row in rows]
    mutation_sets = [set(row["mutations"].split(";")) for row in rows]
    if any(len(mutations) != 10 for mutations in mutation_sets):
        raise ValueError("every design must contain exactly ten mutations")

    # Assign a relative "focus" label so the plate map is useful during triage.
    focus_fields = {
        "stability": "ridgey_stability_delta_mean",
        "EC retention": "ridgey_ec_delta_mean",
        "solubility": "ridgey_solubility_delta_mean",
        "sequence naturalness": "ridgey_masked_lm_delta_mean",
        "evolution/Potts": "potts_delta",
        "fold confidence": "fold_selection_score",
    }
    percentiles = {label: percentile_scores(rows, field) for label, field in focus_fields.items()}
    focus = {
        name: max(focus_fields, key=lambda label: percentiles[label][name])
        for name in names
    }

    overlap = [[len(left & right) for right in mutation_sets] for left in mutation_sets]
    positions = [(row, column) for row in range(6) for column in range(8)]
    edges: list[tuple[int, int, float]] = []
    for index, (row, column) in enumerate(positions):
        for delta_row, delta_column, weight in ((0, 1, 2.0), (1, 0, 2.0), (1, 1, 0.5), (1, -1, 0.5)):
            other = (row + delta_row, column + delta_column)
            if other in positions:
                edges.append((index, positions.index(other), weight))

    metric_fields = [
        "ridgey_stability_delta_mean",
        "ridgey_ec_delta_mean",
        "ridgey_solubility_delta_mean",
        "potts_delta",
        "fold_selection_score",
    ]
    metric_values: list[list[float]] = []
    for field in metric_fields:
        values = [float(row[field]) for row in rows]
        mean = statistics.mean(values)
        sd = statistics.pstdev(values) or 1.0
        metric_values.append([(value - mean) / sd for value in values])

    def plate_cost(order: list[int]) -> float:
        adjacency = 0.0
        for left, right, weight in edges:
            shared = overlap[order[left]][order[right]]
            adjacency += weight * (shared * shared + (20.0 if shared >= 8 else 0.0))
        balance = 0.0
        for values in metric_values:
            for plate_row in range(6):
                offsets = range(plate_row * 8, plate_row * 8 + 8)
                balance += statistics.mean(values[order[offset]] for offset in offsets) ** 2
            for plate_column in range(8):
                offsets = range(plate_column, 48, 8)
                balance += statistics.mean(values[order[offset]] for offset in offsets) ** 2
        return adjacency + 2.5 * balance

    rng = random.Random(SEED)
    current = list(range(48))
    rng.shuffle(current)
    initial_cost = plate_cost(current)
    current_cost = initial_cost
    best = current.copy()
    best_cost = current_cost
    for step in range(100_000):
        left, right = rng.sample(range(48), 2)
        current[left], current[right] = current[right], current[left]
        candidate_cost = plate_cost(current)
        temperature = 6.0 * (0.02 / 6.0) ** (step / 99_999)
        if candidate_cost <= current_cost or rng.random() < math.exp((current_cost - candidate_cost) / temperature):
            current_cost = candidate_cost
            if candidate_cost < best_cost:
                best = current.copy()
                best_cost = candidate_cost
        else:
            current[left], current[right] = current[right], current[left]

    assigned: list[dict] = []
    for offset, design_index in enumerate(best):
        plate_row = ROWS[offset // 8]
        plate_column = offset % 8 + 1
        source = rows[design_index]
        assigned.append({
            "plate": "NGLY1_10M_48",
            "well": f"{plate_row}{plate_column}",
            "row": plate_row,
            "column": plate_column,
            "design": source["design"],
            "design_focus": focus[source["design"]],
            "mutations": source["mutations"],
            "mutation_count": 10,
            "protein_length_aa": len(sequences[source["design"]]),
            "amino_acid_sequence": sequences[source["design"]],
            "ridgey_stability_delta_mean": source["ridgey_stability_delta_mean"],
            "ridgey_stability_delta_min": source["ridgey_stability_delta_min"],
            "ridgey_ec_delta_mean": source["ridgey_ec_delta_mean"],
            "ridgey_ec_non_decreasing_models": source["ridgey_ec_non_decreasing_models"],
            "ridgey_solubility_delta_mean": source["ridgey_solubility_delta_mean"],
            "ridgey_masked_lm_delta_mean": source["ridgey_masked_lm_delta_mean"],
            "potts_delta": source["potts_delta"],
            "esmfold2fast_plddt": source["esmfold2fast_plddt"],
            "esmfold2fast_ptm": source["esmfold2fast_ptm"],
            "core_ca_rmsd_angstrom": source["core_ca_rmsd_angstrom"],
            "minimum_catalytic_ca_distance": source["minimum_catalytic_ca_distance"],
            "structure_file": source["structure_file"],
        })

    map_path = OUT / "ngly1_48well_plate_map.csv"
    with map_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assigned[0]))
        writer.writeheader()
        writer.writerows(assigned)

    layout_path = OUT / "ngly1_48well_plate_layout.csv"
    with layout_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row/column", *COLS])
        for plate_row in ROWS:
            writer.writerow([plate_row, *[next(row["design"] for row in assigned if row["well"] == f"{plate_row}{column}") for column in COLS]])

    fasta_path = OUT / "ngly1_48well_plate_order.fasta"
    with fasta_path.open("w") as handle:
        for row in assigned:
            handle.write(f">{row['well']}|{row['design']} mutations={row['mutations'].replace(';', '|')}\n{row['amino_acid_sequence']}\n")

    adjacent_overlaps = [overlap[best[left]][best[right]] for left, right, weight in edges if weight == 2.0]
    all_pair_overlaps = [len(left & right) for left, right in itertools.combinations(mutation_sets, 2)]
    mutation_usage = Counter(mutation for mutations in mutation_sets for mutation in mutations)
    position_usage = Counter(int(mutation[1:-1]) for mutation in mutation_usage)
    audit = {
        "plate": "NGLY1_10M_48",
        "format": "6 rows x 8 columns",
        "wells": len(assigned),
        "unique_designs": len({row["design"] for row in assigned}),
        "unique_sequences": len({row["amino_acid_sequence"] for row in assigned}),
        "exact_mutations_per_design": 10,
        "unique_mutation_identities": len(mutation_usage),
        "unique_mutated_positions": len(position_usage),
        "median_shared_mutations_all_pairs": statistics.median(all_pair_overlaps),
        "maximum_shared_mutations_all_pairs": max(all_pair_overlaps),
        "minimum_sequence_hamming_distance": 20 - 2 * max(all_pair_overlaps),
        "maximum_shared_mutations_adjacent_wells": max(adjacent_overlaps),
        "median_shared_mutations_adjacent_wells": statistics.median(adjacent_overlaps),
        "layout_optimization_cost_before": initial_cost,
        "layout_optimization_cost_after": best_cost,
        "common_anchor_mutations": mutation_usage.most_common(8),
        "focus_counts": Counter(row["design_focus"] for row in assigned),
        "contains_wt_control": False,
        "control_note": "All 48 wells are ten-mutation designs; place WT and blank controls on a separate assay/control plate if needed.",
    }
    (OUT / "ngly1_48well_plate_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    markdown = [
        "# NGLY1 48-well plate layout",
        "",
        "All wells A1–F8 contain one unique exact-ten-mutation NGLY1 design. Similar mutation sets were dispersed by optimizing orthogonal/diagonal adjacency while balancing Ridgey and folding scores across rows and columns.",
        "",
        "| Row / Column | " + " | ".join(map(str, COLS)) + " |",
        "|---|" + "---|" * 8,
    ]
    for plate_row in ROWS:
        markdown.append(
            f"| {plate_row} | "
            + " | ".join(next(row["design"] for row in assigned if row["well"] == f"{plate_row}{column}") for column in COLS)
            + " |"
        )
    markdown.extend([
        "",
        f"Diversity: {audit['unique_mutation_identities']} mutation identities across {audit['unique_mutated_positions']} positions; median pairwise shared mutations {audit['median_shared_mutations_all_pairs']:.1f}/10; no pair shares more than {audit['maximum_shared_mutations_all_pairs']}/10.",
        "",
        "This is an all-mutant plate. WT, empty-vector, and blank controls are not included because the requested 48 wells are occupied by the 48 designs.",
    ])
    (OUT / "ngly1_48well_plate_layout.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
