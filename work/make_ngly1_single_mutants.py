#!/usr/bin/env python3
"""Create the union of single mutations represented in the ten-mutant pool."""

import csv
from pathlib import Path


ROOT = Path("/home/ubuntu/codex_ngly1_20260819")


def read_wt() -> str:
    return "".join(
        line.strip()
        for line in (ROOT / "ngly1.fasta").read_text().splitlines()
        if not line.startswith(">")
    )


def main() -> None:
    wt = read_wt()
    mutations: set[str] = set()
    with (ROOT / "work/ngly1_10mut_pool.csv").open() as handle:
        for row in csv.DictReader(handle):
            mutations.update(row["mutations"].split(";"))
    records: list[str] = []
    for mutation in sorted(mutations, key=lambda text: (int(text[1:-1]), text[-1])):
        native, position, target = mutation[0], int(mutation[1:-1]), mutation[-1]
        if wt[position - 1] != native:
            raise ValueError(f"WT mismatch for {mutation}")
        sequence = wt[: position - 1] + target + wt[position:]
        records.extend([f">{mutation}", sequence])
    output = ROOT / "work/ngly1_union_single_mutants.fasta"
    output.write_text("\n".join(records) + "\n")
    print(f"wrote {len(mutations)} single mutants to {output}")


if __name__ == "__main__":
    main()
