# Code Basis Map

This file maps the collected public source artifacts onto the implementation tasks in [agent.md](../agent.md).

## Collected papers

- `2310.18864v1`, posted on October 29, 2023
  `data/external/arxiv/2310.18864v1/`
- `2502.09044`, posted on February 13, 2025
  `data/external/arxiv/2502.09044/`
- `2503.10089`, posted on March 13, 2025
  `data/external/arxiv/2503.10089/`

The following papers are registered in the code but were not fetched by default because their public arXiv source bundles do not expose ancillary table directories:

- `2411.09132`, posted on November 14, 2024
- `2508.20995`, posted on August 28, 2025

## Task-to-data mapping

### 1. Wilson line constrained extraction

Primary basis:
- `data/external/arxiv/2310.18864v1/anc/README.txt`
- `data/external/arxiv/2310.18864v1/anc/Fig2_195.dat`
- `data/external/arxiv/2310.18864v1/anc/Fig2_251.dat`
- `data/external/arxiv/2310.18864v1/anc/Fig2_293.dat`
- `data/external/arxiv/2310.18864v1/anc/Fig2_352.dat`
- `data/external/arxiv/2310.18864v1/anc/Fig3.dat`
- `data/external/arxiv/2310.18864v1/anc/Fig5.dat`
- `data/external/arxiv/2310.18864v1/anc/Fig6.dat`

Why these matter:
- `Fig2_*` gives first-cumulant benchmarks `m1(r, tau, T)` at several temperatures.
- `Fig3.dat` gives the interference function `phi(r, T)`.
- `Fig5.dat` gives the extracted potential `V(r, T)`.
- `Fig6.dat` gives the resulting static `Q Qbar` spectral functions.

Direct code hooks:
- load with `hefty_tm.datasets.load_table`
- interpolate `phi(r)` and `V(r)` with `hefty_tm.spectral.TabulatedRadialProfile`
- reconstruct or sanity-check `rho_QQbar`, `W(r, tau, T)`, and `m1`

### 2. Spectral functions and pole structure

Primary basis:
- `data/external/arxiv/2502.09044/anc/Figure1.dat`
- `data/external/arxiv/2502.09044/anc/Figure4_lines_1S.dat`
- `data/external/arxiv/2502.09044/anc/Figure4_lines_2S.dat`
- `data/external/arxiv/2502.09044/anc/Figure4_lines_3S.dat`
- `data/external/arxiv/2502.09044/anc/Figure4_lines_4S.dat`
- `data/external/arxiv/2502.09044/anc/Figure5_195MeV.dat`
- `data/external/arxiv/2502.09044/anc/Figure5_293MeV.dat`
- `data/external/arxiv/2502.09044/anc/Figure5_500MeV.dat`

Why these matter:
- `Figure1.dat` is a direct complex-energy-plane table for `|T|^2`.
- `Figure4_lines_*` tracks spectroscopy information by state label.
- `Figure5_*` provides temperature-resolved widths or dissociation-related benchmarks used in the spectroscopy analysis.

Notes:
- The exact semantic meaning of some `Figure4_*` and `Figure5_*` columns should be checked against the paper captions before coding a production pole finder.
- That interpretation is straightforward to add later by parsing the paper source, but the public table basis is already present locally.

### 3. Dissociation-rate construction

Primary basis:
- `data/external/arxiv/2503.10089/anc/Fig13a_cc_Jpsi_T194_SCS_onshell.txt`
- `data/external/arxiv/2503.10089/anc/Fig14a_Y1S_T194_SCS_onshell.txt`
- `data/external/arxiv/2503.10089/anc/Fig16a_Y1S_T195_WLC_withInterf.txt`
- `data/external/arxiv/2503.10089/anc/Fig17.tsv`
- `data/external/arxiv/2503.10089/anc/Fig19a_Y1S_T195_Full.txt`
- `data/external/arxiv/2503.10089/anc/Fig20a_JPsi_T195_Full.txt`

Why these matter:
- the `Fig13*` and `Fig14*` families compare different rate constructions for charmonium and bottomonium
- the `Fig16*` family isolates the impact of interference effects
- `Fig17.tsv` is a compact tabular artifact suitable for direct loader tests
- the `Fig19*` and `Fig20*` families break rates into components; this is useful for validating any future rate-equation source term decomposition

Notes:
- For the large `2503.10089` asset tree, the filename itself carries a useful state / temperature / model encoding.
- The current loader keeps that structure intact instead of flattening or renaming files.

### 4. Rate-equation handoff

Current local basis:
- `src/hefty_tm/rates.py`
- `src/hefty_tm/wilson_line.py`
- `data/external/arxiv/2503.10089/anc/Fig19*.txt`
- `data/external/arxiv/2503.10089/anc/Fig20*.txt`

What is implemented now:
- thermal weighting helper
- weighted spectral integral helper
- explicit rate-equation time stepping

What still needs a dedicated implementation:
- state-resolved in-medium dissociation and regeneration kernels
- coupling to a medium evolution model
- calibration against the transport paper `2508.20995`

## Local commands

List papers:

```bash
PYTHONPATH=src python3 -m hefty_tm list-papers
```

Refetch public assets:

```bash
PYTHONPATH=src python3 -m hefty_tm fetch
```

Summarize any fetched asset directory:

```bash
PYTHONPATH=src python3 -m hefty_tm summarize-data data/external/arxiv/2502.09044
```
