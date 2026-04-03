# Code Structure and Current Workflow

Last synchronized with code on 2026-04-03.

This is the canonical implementation note for the local HEFTY workspace. It
describes what the code actually does today, which commands are runnable, and
how the current task-1 branches differ. If another note disagrees with this
file, trust this file.

## Scope

The repository is a reduced-static benchmark scaffold, not the full HEFTY
thermodynamic T-matrix code. In particular, the current codebase still does not
contain:

- the full light-parton / heavy-light outer self-consistency loop
- a complete causal heavy-quark self-energy construction from the medium sector
- the full bottomonium pole-search and rate workflow from the later papers

What it does contain is a set of public-data benchmark branches for the static
Wilson-line sector, plus utilities for fetching ancillary data, loading tables,
and producing the analysis figures used in the Overleaf note.

## Workspace Layout

- `src/hefty_tm/`
  Python package with the CLI, data loaders, fetch helpers, and the task-1
  benchmark implementation.
- `data/external/arxiv/`
  Fetched arXiv source bundles and ancillary tables.
- `lattice/benchmark_wlc/`
  Curated public Wilson-line benchmark inputs.
- `results/`
  Saved benchmark outputs for the different task-1 branches.
- `overleaf_69cda82ae8be36b19f2caff6/`
  Analysis note and synced figures for the Overleaf project.
- `tests/`
  Lightweight regression tests.

## Python Package Structure

### CLI and entry points

- `src/hefty_tm/__main__.py`
  Package entry point.
- `src/hefty_tm/cli.py`
  Defines the command-line interface.

The CLI now lazy-imports the benchmark runner module. This means lightweight
commands such as `list-papers`, `fetch`, `summarize-data`, and `demo-wlc` do not
eagerly import `matplotlib` or `scipy`.

### Data and paper utilities

- `src/hefty_tm/papers.py`
  Paper registry and fetch defaults.
- `src/hefty_tm/fetch.py`
  ArXiv download and ancillary extraction helpers. Extraction now rejects path
  traversal entries.
- `src/hefty_tm/datasets.py`
  Generic loaders for `.dat`, `.txt`, and `.tsv` tables. Headerless one-column
  numeric files are supported.

### Physics helpers

- `src/hefty_tm/spectral.py`
  Minimal spectral-function utilities.
- `src/hefty_tm/wilson_line.py`
  Wilson-line Laplace transforms, cumulants, and `tau -> 0` extrapolation
  helpers.
- `src/hefty_tm/rates.py`
  Simple rate-equation stepping utilities.

### Task-1 benchmark implementation

- `src/hefty_tm/benchmark_task1.py`
  The operational heart of the current static Wilson-line benchmark workflow.

## Runtime Commands

Current CLI commands are:

- `list-papers`
- `fetch`
- `summarize-data`
- `demo-wlc`
- `task1-benchmark`
- `task1-publication-faithful`
- `task1-publication-locked`
- `task1-publication-smoothed`
- `task1-tang-exact`

## Task-1 Branch Map

There are several task-1 branches on purpose. They solve different inverse
problems.

### 1. `task1-benchmark`

- Calls `run_task1_benchmark(...)`.
- Default output directory: `results/task1`
- This is the branch currently treated as the main saved benchmark and the one
  synced into the Overleaf note.

### 2. `task1-publication-faithful`

- Also calls `run_task1_benchmark(...)`.
- Default output directory: `results/task1_publication_faithful`
- This is a semantic alias of `task1-benchmark`; only the default output
  directory differs.

### 3. `task1-publication-locked`

- Calls `run_task1_publication_locked_benchmark(...)`.
- Default output directory: `results/task1_publication_locked_potential`
- This branch hard-locks the screened-Cornell backbone to Tang Fig. 4 and then
  uses extra reduced-static corrections to improve figure-level agreement.

### 4. `task1-publication-smoothed`

- Calls `run_task1_publication_smoothed_benchmark(...)`.
- Default output directory: `results/task1_publication_smoothed_fig4`
- This branch replaces the discrete Fig. 4 table by a smoothed temperature-law
  surrogate and is mainly a diagnostic/experiment branch.

### 5. `task1-tang-exact`

- Calls `run_task1_tang_exact_benchmark(...)`.
- Default output directory: `results/task1_tang_exact`
- This branch is a Tang-style replay diagnostic. It removes the hybrid
  publication-faithful penalties and fixes the interference function and kernel
  closer to the original Tang setup.

## What the Main Saved Branch Actually Fits

The main saved branch, `run_task1_benchmark(...)`, is a publication-faithful
reduced-static hybrid fit. It is not a literal reproduction of Tang 2024.

Its objective mixes:

- the public subtracted Euclidean `m1(r, tau, T)` curves
- weak Tang Fig. 4 parameter priors
- direct Tang Fig. 5 potential residuals
- Tang Fig. 6 spectral-shape and summary residuals
- public `c1(r,T)` validation terms
- a model-inspired outer-anchor prior from the public `WLC -> SCS` difference

This is why the current branch should be read as a different inverse problem
from Tang’s original extraction, not as a one-to-one parameter replay.

The main fit degrees of freedom in this branch include:

- `m_d(T)`
- a nearly common `m_s`
- `c_b(T)`
- `phi(0.224,T)`, `phi(0.505,T)`, and `phi(0.757,T)`
- a reduced energy/radius parameterization of `Sigma_{Q\\bar Q}(E,T)`

Important audit result:

- `phi(0.757,T)` is an actual independent fit parameter in the code. It is not
  a derived display-only quantity.

## What the Tang-Exact Branch Actually Fits

The `task1-tang-exact` branch is the cleaner diagnostic branch introduced to
separate structural disagreement from implementation drift.

Relative to the main publication-faithful branch, it:

- fixes `phi(r,T)` to the Tang Fig. 3 reference
- fixes `m_s = 0.2 GeV`
- fixes the self-energy kernel to the Fig. 6-inferred reference kernel
- removes the direct Tang Fig. 5 / Fig. 6 penalties
- removes the public `c1(r,T)` term
- removes the `WLC -> SCS` outer-anchor prior

It is therefore much closer to Tang’s original Wilson-line inverse problem, but
it also fits the current public Euclidean benchmark much worse. That is an
important diagnostic result, not a bug.

## Current Figure and Output Provenance

The Overleaf note is intended to use one consistent main branch at a time. The
current synced figure set comes from the saved `results/task1/` branch unless a
figure caption explicitly states that a different diagnostic branch is being
shown.

Typical outputs written by a task-1 branch include:

- `fit_params.json`
- `report.md`
- `plot_m1_reproduction.png`
- `plot_Vtilde_extracted.png`
- `plot_phi_extracted.png`
- `plot_spectral_extraction.png`
- `plot_spectral_ansatz_sensitivity.png`
- `plot_public_c1_validation.png`
- `plot_public_wilson_validation.png`
- `plot_fit_performance.png`

## Data Inputs Used by the Current Workflows

### Primary public benchmark inputs

- public subtracted `m1(r, tau, T)` tables in `lattice/benchmark_wlc/`
- public ancillary reference tables from `data/external/arxiv/2310.18864v1/anc/`

### Important caveat

The exact line-by-line finite-temperature raw Wilson-line correlators for the
benchmark temperatures are not public in the same form used by Tang. The code
therefore works from public derived benchmark products, not from an exact replay
of the original raw-data pipeline.

That difference matters scientifically. Small changes in subtraction,
interpolation, or the `tau` window can shift the extracted `tau -> 0` intercepts
and therefore move the effective potential parameters.

## Validation and Testing

The test suite is in `tests/test_core.py`.

Current regression coverage includes:

- Wilson-line helper utilities
- ancillary-table loading, including one-column numeric files
- the safe ancillary extraction path checks
- benchmark data loaders
- Tang Fig. 4 / Fig. 5 consistency checks

Tests now use the project root relative to the checkout and no longer hard-code
`/raid5/data/yjlee/hefty`.

Run validation with:

```bash
python3 -m py_compile src/hefty_tm/cli.py src/hefty_tm/fetch.py src/hefty_tm/datasets.py tests/test_core.py
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Practical Reading Guide

If you need one short map:

- use `README.md` for quick start
- use this file for the current truth about the code
- use `task.md` and `agent.md` only for historical planning context
- use `docs/publication_gap.md` only as a historical branch-comparison note
- use the Overleaf note for the current narrative presentation of the saved
  branch results
