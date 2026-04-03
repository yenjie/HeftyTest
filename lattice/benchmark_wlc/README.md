# Wilson-Line Benchmark Subset

This directory is the first clean lattice subset for the benchmark defined in [task.md](../../task.md):

- reproduce Tang et al. Fig. 2 for `m1(r,tau,T)`
- use the `tau -> 0` intercept to extract `V_tilde(r,T)` as in Tang et al. Fig. 5

## Source package

All files here were selected from:

- [../bazavov_2308.16587v2](../bazavov_2308.16587v2)

which comes from the ancillary `data.tar.xz` archive of:

- Bazavov et al., `Un-screened forces in Quark-Gluon Plasma?`
  arXiv:2308.16587v2

## Benchmark mapping

The four benchmark temperatures are taken from the single `beta = 8.249` lattice spacing:

- `Nt = 36 -> T = 0.195 GeV`
- `Nt = 28 -> T = 0.251 GeV`
- `Nt = 24 -> T = 0.293 GeV`
- `Nt = 20 -> T = 0.352 GeV`

Using `a = 0.1973269804 / (Nt * T)` in fm gives a nearly constant lattice spacing,
`a ~= 0.0281 fm`, so the Tang benchmark distances map to:

- `r/a = 8 -> r ~= 0.224 fm`
- `r/a = 18 -> r ~= 0.505 fm`
- `r/a = 27 -> r ~= 0.757 fm`

## Selected files

Primary benchmark inputs:

- `data_subtracted/`

These are the already-subtracted effective-mass files and are the right default choice for the first Tang benchmark, since [task.md](../../task.md) explicitly says not to fit raw short-`tau` data with unresolved excited-state contamination.

Support files:

- `data_raw/`
  matching non-subtracted Bazavov `Meff` files
- `data_fit/`
  matching Bazavov fit curves for the same temperatures and distances
- `reference_exports/`
  a few figure-level exports from Bazavov's `plots_data/` directory

## Column conventions

Observed file layout:

- `data_subtracted/*.txt`: `tauT`, `m1`, `sigma_m1`
- `data_raw/*.txt`: `tauT`, `m1`, `sigma_m1`
- `data_fit/*.txt`: `tauT`, `fit_curve`

This is inferred directly from file contents and naming, and should be validated again when the fitting pipeline is wired.

## Notes

- Bazavov's `plots_data/` exports are incomplete for the exact Tang temperature set.
- In particular there is no direct `figure2_T195.txt` or `figure2_T293.txt` export.
- That is why this subset is built from `data_plots_final/` rather than from `plots_data/`.

See `manifest.json` for the exact file mapping back to the original ancillary package.
