import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio.Align import substitution_matrices
from Bio.PDB import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley


ENSEMBLE_PATH = Path("outputs/ridgey_600m_ensemble_raw.json")
MSA_PATH = Path("outputs/ngly1_msa.a3m")
EV_PATH = Path("outputs/ngly1_ev.npz")
STRUCTURE_PATH = Path("work/ngly1_afdb.cif")
POOL_FASTA = Path("work/ngly1_10mut_pool.fasta")
POOL_CSV = Path("work/ngly1_10mut_pool.csv")
POOL_SIZE = 160
SEED = 20260819

AA = "ACDEFGHIKLMNPQRSTVWY"
MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
    "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
    "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}
HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2,
    "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}


def read_a3m(path):
    rows, current = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                rows.append(re.sub("[a-z.]", "", "".join(current)))
                current = []
        else:
            current.append(line.strip())
    if current:
        rows.append(re.sub("[a-z.]", "", "".join(current)))
    return rows


def potts_score(states, h, v):
    selected_v = v[np.arange(len(states)), :, states]
    field = h[np.arange(len(states)), states].sum()
    total = selected_v.sum(axis=0)
    pair = 0.5 * ((total * total).sum() - (selected_v * selected_v).sum())
    return float(field + pair)


def main():
    rng = random.Random(SEED)
    ensemble = json.loads(ENSEMBLE_PATH.read_text())[0]
    sequence = ensemble["sequence"]
    predictions = ensemble["predictions"]
    plddt = ensemble["structure_confidence"]["plddt"]
    msa = [row for row in read_a3m(MSA_PATH) if len(row) == len(sequence)]
    ev = np.load(EV_PATH)
    ev_aa = str(ev["aa_order"])
    h, v = ev["h"], ev["V"]
    query_states = np.array([1 + ev_aa.index(amino_acid) for amino_acid in sequence])
    wild_type_potts = potts_score(query_states, h, v)

    structure = MMCIFParser(QUIET=True).get_structure("ngly1", str(STRUCTURE_PATH))
    ShrakeRupley().compute(structure, level="R")
    residues = [residue for residue in next(structure.get_chains()) if residue.id[0] == " "]
    ca = np.array([residue["CA"].coord for residue in residues])

    # Strong catalytic protection: freeze the whole catalytic block, every
    # residue within 18 A CA distance of the catalytic/Zn sites, and every
    # Ridgey-predicted partner-contact residue within 8 A.
    protected = set(range(112, 169)) | set(range(240, 381)) | {2, 137}
    catalytic_and_zinc = [250, 253, 283, 286, 309, 336, 353]
    for position in range(1, len(sequence) + 1):
        if min(np.linalg.norm(ca[position - 1] - ca[site - 1]) for site in catalytic_and_zinc) < 18:
            protected.add(position)
    for prediction in predictions:
        for head in ("anything_8", "ligand_8"):
            protected.update(
                position + 1
                for position in prediction["contact"]["positions_0_indexed"].get(head, [])
            )

    blosum = substitution_matrices.load("BLOSUM62")
    mutation_pool = []
    for index, wild_type in enumerate(sequence):
        position = index + 1
        if position in protected or plddt[index] < 85:
            continue
        counts = Counter(row[index] for row in msa if row[index] in AA)
        non_gap = sum(counts.values())
        if non_gap < 300:
            continue
        wild_type_frequency = counts[wild_type] / non_gap
        relative_sasa = residues[index].sasa / MAX_ASA[wild_type]
        max_ptm = max(prediction["ptm"]["any_probability"][index] for prediction in predictions)
        if max_ptm >= 0.25:
            continue
        for mutant in AA:
            if mutant == wild_type or mutant in "PGC":
                continue
            mutant_frequency = counts[mutant] / non_gap
            if mutant_frequency < 0.005:
                continue
            improvements, lm_deltas = [], []
            for prediction in predictions:
                ddg_order = prediction["ddg_amino_acid_order"]
                lm_order = prediction["lm_amino_acid_order"]
                improvements.append(
                    prediction["ddg"][index][ddg_order.index(wild_type)]
                    - prediction["ddg"][index][ddg_order.index(mutant)]
                )
                lm_deltas.append(
                    prediction["lm"][index][lm_order.index(mutant)]
                    - prediction["lm"][index][lm_order.index(wild_type)]
                )
            mean_improvement = statistics.mean(improvements)
            mean_lm_delta = statistics.mean(lm_deltas)
            if min(improvements) <= 0 or mean_improvement < 0.15:
                continue
            if sum(value > 0 for value in lm_deltas) < 3 or mean_lm_delta <= -0.5:
                continue
            blosum_score = float(blosum[wild_type, mutant])
            if blosum_score < -2 and not (
                mutant_frequency >= 0.08 and mutant_frequency >= wild_type_frequency
            ):
                continue
            if relative_sasa > 0.35 and HYDROPATHY[mutant] > HYDROPATHY[wild_type] + 1.0:
                continue
            if relative_sasa < 0.20 and mutant in "DEKR" and mutant_frequency < 0.10:
                continue
            independent_delta = float(ev["dE_independent"][index, ev_aa.index(mutant)])
            potts_delta = float(ev["dE_potts"][index, ev_aa.index(mutant)])
            # Ten substitutions cannot all be Potts-positive for this human
            # sequence. Exclude only the strongly implausible tail here, then
            # rank complete designs by their exact multi-mutant Potts score.
            if potts_delta < -8.0 or independent_delta < -3.5:
                continue
            mutation_pool.append({
                "mutation": f"{wild_type}{position}{mutant}",
                "position": position,
                "wild_type": wild_type,
                "mutant": mutant,
                "improvements": improvements,
                "mean_improvement": mean_improvement,
                "min_improvement": min(improvements),
                "lm_deltas": lm_deltas,
                "mean_lm_delta": mean_lm_delta,
                "mutant_frequency": mutant_frequency,
                "wild_type_frequency": wild_type_frequency,
                "independent_delta": independent_delta,
                "potts_single_delta": potts_delta,
                "relative_sasa": relative_sasa,
                "plddt": plddt[index],
                "charge_delta": CHARGE.get(mutant, 0) - CHARGE.get(wild_type, 0),
                "surface_hydropathy_delta": (
                    HYDROPATHY[mutant] - HYDROPATHY[wild_type]
                    if relative_sasa > 0.35 else 0.0
                ),
            })

    by_position = defaultdict(list)
    for mutation in mutation_pool:
        by_position[mutation["position"]].append(mutation)

    positions = sorted(by_position)
    designs = {}
    for _ in range(500_000):
        if len(positions) < 10:
            raise RuntimeError(f"Only {len(positions)} mutable positions survived filtering")
        chosen_positions = rng.sample(positions, 10)
        chosen = []
        for position in chosen_positions:
            options = by_position[position]
            weights = [
                math.exp(
                    min(4.0, option["mean_improvement"])
                    + 0.10 * option["mean_lm_delta"]
                    + 0.06 * option["potts_single_delta"]
                )
                for option in options
            ]
            chosen.append(rng.choices(options, weights=weights, k=1)[0])

        # Avoid directly interacting mutations and extreme charge shifts.
        if any(
            np.linalg.norm(ca[a["position"] - 1] - ca[b["position"] - 1]) < 6.0
            for i, a in enumerate(chosen)
            for b in chosen[i + 1:]
        ):
            continue
        net_charge_delta = sum(option["charge_delta"] for option in chosen)
        if abs(net_charge_delta) > 2:
            continue
        surface_hydropathy_delta = sum(option["surface_hydropathy_delta"] for option in chosen)
        if surface_hydropathy_delta > 1.0:
            continue
        # Keep the changes distributed rather than redesigning only the PAW domain.
        if sum(option["position"] >= 454 for option in chosen) > 7:
            continue
        if sum(option["position"] < 454 for option in chosen) < 2:
            continue

        additive_by_model = [
            sum(option["improvements"][model_index] for option in chosen)
            for model_index in range(5)
        ]
        lm_by_model = [
            sum(option["lm_deltas"][model_index] for option in chosen)
            for model_index in range(5)
        ]
        if min(additive_by_model) < 2.5 or min(lm_by_model) < -1.0:
            continue

        states = query_states.copy()
        for option in chosen:
            states[option["position"] - 1] = 1 + ev_aa.index(option["mutant"])
        potts_delta = potts_score(states, h, v) - wild_type_potts
        independent_delta = sum(option["independent_delta"] for option in chosen)

        mutations = tuple(sorted((option["mutation"] for option in chosen), key=lambda x: int(x[1:-1])))
        score = (
            0.7 * statistics.mean(additive_by_model)
            + 0.6 * min(additive_by_model)
            + 0.08 * statistics.mean(lm_by_model)
            + 0.04 * potts_delta
            + 0.03 * independent_delta
            - 0.05 * abs(net_charge_delta)
        )
        existing = designs.get(mutations)
        if existing is None or score > existing["generation_score"]:
            designs[mutations] = {
                "mutations": mutations,
                "chosen": chosen,
                "generation_score": score,
                "additive_mean": statistics.mean(additive_by_model),
                "additive_min": min(additive_by_model),
                "lm_mean": statistics.mean(lm_by_model),
                "lm_min": min(lm_by_model),
                "potts_delta": potts_delta,
                "independent_delta": independent_delta,
                "net_charge_delta": net_charge_delta,
                "surface_hydropathy_delta": surface_hydropathy_delta,
            }

    ranked = sorted(designs.values(), key=lambda design: design["generation_score"], reverse=True)
    selected = []
    mutation_use = Counter()
    for design in ranked:
        mutation_set = set(design["mutations"])
        if any(len(mutation_set & set(other["mutations"])) > 8 for other in selected):
            continue
        if any(mutation_use[mutation] >= int(0.85 * POOL_SIZE) for mutation in mutation_set):
            continue
        selected.append(design)
        mutation_use.update(mutation_set)
        if len(selected) == POOL_SIZE:
            break
    if len(selected) < POOL_SIZE:
        raise RuntimeError(f"Only selected {len(selected)} diverse designs")

    fieldnames = [
        "name", "mutations", "generation_score", "additive_mean",
        "additive_min", "lm_mean", "lm_min", "potts_delta",
        "independent_delta", "net_charge_delta", "surface_hydropathy_delta",
        "minimum_catalytic_ca_distance",
    ]
    with POOL_CSV.open("w", newline="") as handle, POOL_FASTA.open("w") as fasta:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for design_index, design in enumerate(selected, 1):
            name = f"NGLY1_D{design_index:03d}"
            designed_sequence = list(sequence)
            for option in design["chosen"]:
                designed_sequence[option["position"] - 1] = option["mutant"]
            designed_sequence = "".join(designed_sequence)
            minimum_catalytic_distance = min(
                np.linalg.norm(ca[option["position"] - 1] - ca[site - 1])
                for option in design["chosen"]
                for site in catalytic_and_zinc
            )
            writer.writerow({
                "name": name,
                "mutations": ";".join(design["mutations"]),
                "generation_score": design["generation_score"],
                "additive_mean": design["additive_mean"],
                "additive_min": design["additive_min"],
                "lm_mean": design["lm_mean"],
                "lm_min": design["lm_min"],
                "potts_delta": design["potts_delta"],
                "independent_delta": design["independent_delta"],
                "net_charge_delta": design["net_charge_delta"],
                "surface_hydropathy_delta": design["surface_hydropathy_delta"],
                "minimum_catalytic_ca_distance": minimum_catalytic_distance,
            })
            fasta.write(f">{name} mutations={'|'.join(design['mutations'])}\n{designed_sequence}\n")

    Path("work/ngly1_protected_positions.json").write_text(
        json.dumps({
            "protected_positions_1_indexed": sorted(protected),
            "catalytic_positions": [309, 336, 353],
            "zinc_ligands": [250, 253, 283, 286],
            "catalytic_shell_ca_cutoff_angstrom": 18.0,
            "candidate_mutations": mutation_pool,
        }, indent=2) + "\n"
    )
    print(json.dumps({
        "eligible_mutations": len(mutation_pool),
        "eligible_positions": len(by_position),
        "unique_designs_generated": len(designs),
        "selected_pool": len(selected),
        "top_score": selected[0]["generation_score"],
        "minimum_score": selected[-1]["generation_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
