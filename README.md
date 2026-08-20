# NGLY1 ten-mutation redesign

This repository contains a computationally screened set of 48 exact ten-substitution variants of human NGLY1 (UniProt Q96IV0, EC 3.5.1.52).

The finished deliverable is in `outputs/ngly1_48_10mut_designs/`:

- `ngly1_48_10mut_designs.fasta` — the 48 sequences
- `ngly1_48_10mut_designs.csv` — per-design Ridgey, Potts, solubility, masked-PPL, and folding audit columns
- `ngly1_48_manifest.json` — method and gate definitions
- `structures/` and `ngly1_48_structures.zip` — ESMFold2-Fast mmCIF predictions
- `ngly1_48well_plate_map.csv` — the complete A1–F8 plate/order sheet
- `ngly1_48well_plate_layout.csv` and `.md` — compact 6×8 grid views
- `ngly1_48well_plate_order.fasta` — sequences ordered by well

## Design screen

1. A 5,000-sequence MMseqs MSA was fit with a rank-32 low-rank Potts model using an eight-value L2 sweep; the selected regularization was λ=1.0.
2. Catalytic residues 309/336/353, Zn ligands 250/253/283/286, the complete 18 Å CA shell around them, Ridgey contact sites, PTM sites, and low-confidence residues were frozen.
3. Every proposed substitution was observed in the MSA. Complete ten-mutation sequences were scored, not only summed single-mutant predictions.
4. Five local Ridgey v2 600M checkpoints screened stability, EC 3.5.1.52, enzyme/Zn calls, solubility, active sites, and sequence-only mutation-site masked perplexity.
5. The top 64 consensus designs were folded locally with ESMFold2-Fast (three loops, 50 sampling steps); 48 were selected using pLDDT, pTM, and per-domain CA RMSD.

All inference and folding runs were executed on `aws0`; model checkpoints and large intermediates are intentionally excluded from git.

These are computational candidates, not experimentally validated stabilized enzymes. Expression, activity, Zn loading, and thermal-shift measurements remain necessary.
