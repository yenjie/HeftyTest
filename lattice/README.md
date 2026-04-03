# Lattice Data

This directory holds the public lattice-QCD data package currently selected for the first benchmark in [task.md](../task.md).

## Source

Primary lattice-side companion paper:

- Alexei Bazavov, Daniel Hoying, Rasmus N. Larsen, Swagato Mukherjee, Peter Petreczky, Alexander Rothkopf, Johannes Heinrich Weber,
  `Un-screened forces in Quark-Gluon Plasma?`
  arXiv:2308.16587v2
  https://arxiv.org/abs/2308.16587

The arXiv abstract page exposes an ancillary archive:

- `data.tar.xz`
  https://arxiv.org/src/2308.16587v2/anc/data.tar.xz

## Local files

Downloaded archive:

- `bazavov_2308.16587v2_data.tar.xz`

Extracted package:

- `bazavov_2308.16587v2/data/readme.txt`
- `bazavov_2308.16587v2/data/Paper_fits.ipynb`
- `bazavov_2308.16587v2/data/data_plots_final/`
- `bazavov_2308.16587v2/data/plots_data/`

## Package structure

Per the package's own `readme.txt`:

- `data_plots_final/` contains the data used to generate the paper plots
- `plots_data/` contains figure-level outputs written by `Paper_fits.ipynb`

## Benchmark-relevant notes

For the Tang Fig. 2 / Fig. 5 benchmark, this package is the best public lattice-side starting point because it contains temporal Wilson-line analysis data and already-subtracted effective-mass style files.

Direct figure-level exports present in `plots_data/` include:

- `figure2_T195.txt`
- `figure2_T251.txt`
- `figure2_T352.txt`
- `figure3_right_T293.txt`

There is no direct `figure2_T293.txt` export in `plots_data/`. The `293 MeV` benchmark point will need to be traced either through the notebook logic in `Paper_fits.ipynb` or through the lower-level files in `data_plots_final/`.

Likely raw or processed benchmark inputs live in `data_plots_final/` under names such as:

- `Meff_*_sub_*`
- `c1_*`
- `sc2_*`
- `values_*`

Those will be the next files to inspect when wiring the first benchmark dataset.
