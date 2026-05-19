# Wolfram Audit Scripts

This directory contains optional Wolfram Language audit scripts for the Higgs
module. They are not required for the Python regeneration path, and they are
not treated as evidence by themselves. Their role is to make selected symbolic
and algebraic checks independently inspectable in Mathematica/Wolfram Language.

Run them, when `wolframscript` is available, with:

```bash
wolframscript -file wolfram/Higgs_Quartic_Overlap_Verification.wl
wolframscript -file wolfram/BranchA_Stabilizer_Hypercharge_Audit.wl
wolframscript -file wolfram/Projection_BRST_Skeleton.wl
```

Expected scope:

- `Higgs_Quartic_Overlap_Verification.wl` checks the normalized
  `sech^nu(x)` zero mode, the quartic overlap formula, the `nu=3/10` numerical
  value, and the sensitivity table.
- `BranchA_Stabilizer_Hypercharge_Audit.wl` checks the `3+2` generator,
  canonical normalization, and `T_Sigma = -T_Y`.
- `Projection_BRST_Skeleton.wl` checks only the algebraic skeleton:
  `s^2 H = 0`, `Q h = 0`, and the factorized operator form. It is not an
  anomaly or regulator proof.

The main manuscript remains cautious: these scripts support reproducibility of
the formal skeleton, not validation of the full physical theory.
