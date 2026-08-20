# NGLY1 48 × ten-mutation design set

This directory contains 48 exact ten-substitution variants of human NGLY1 (Q96IV0; 654 aa). These are computational candidates, not experimentally validated proteins.

## Hard constraints used

- No substitution occurs in the catalytic triad (309, 336, 353), the Zn ligands (250, 253, 283, 286), the entire 18 Å CA shell around those sites, or Ridgey contact/PTM protection masks.
- Stability improves against WT in every one of five local Ridgey v2 600M replicas.
- Mean P(EC 3.5.1.52) improves; at least four of five replicas are non-decreasing, with the single-replica tolerance capped at 0.0018.
- Mean solubility and sequence-only mutation-site masked log-probability improve.
- Every substitution is observed in the 5,000-sequence MSA and complete designs were ranked by the rank-32, L2-regularized Potts model (selected λ=1.0).
- ESMFold2-Fast predicts WT-like folding using three recycling loops and 50 sampling steps.

## Final-set ranges

- Ridgey stability Δ vs WT, ensemble mean: 0.113 to 0.221 kcal/mol-equivalent scale
- Worst-replica Ridgey stability Δ: 0.093 to 0.210
- Mean P(EC 3.5.1.52) Δ: 0.000841 to 0.002562
- Mean solubility Δ: 0.0007 to 0.0236
- ESMFold2-Fast pLDDT: 0.9324 to 0.9395 (WT 0.9342)
- ESMFold2-Fast pTM: 0.5946 to 0.6250 (WT 0.5974)
- Catalytic-core CA RMSD to the WT prediction: 0.043 to 0.156 Å

Use `ngly1_48_10mut_designs.csv` for per-design audit columns and `structures/` for the predicted mmCIF files. Experimental expression, activity, metal loading, and thermal-shift testing are still required before calling any design stabilized in the laboratory.

## Plate-ready files

- `ngly1_48well_plate_map.csv` maps every design to one well in a 6×8 plate and includes its amino-acid sequence and key screening metrics.
- `ngly1_48well_plate_layout.csv` and `.md` provide compact grid views.
- `ngly1_48well_plate_order.fasta` is ordered A1 through F8.
- `ngly1_48well_plate_audit.json` records the plate and diversity invariants.

The layout disperses closely related designs and balances the scoring profiles across rows and columns. All 48 wells contain mutants; WT and blank controls require a separate control plate.
