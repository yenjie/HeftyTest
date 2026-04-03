# HEFTY T-Matrix Starter Basis

This repository is a local code basis for a lattice-QCD-informed reduced-static
T-matrix benchmark. The canonical implementation note is
[docs/code_structure.md](docs/code_structure.md), which is synchronized with the
current code on 2026-04-03.

The short planning notes [task.md](task.md), [agent.md](agent.md), and
[docs/publication_gap.md](docs/publication_gap.md) are kept as historical or
supplementary context only. They should not be treated as the source of truth
for the current runtime workflow.

The workspace originally only contained literature notes. This scaffold now
contains:

- a paper registry for the HEFTY / Ralf Rapp extraction chain
- a fetch layer for arXiv source packages and ancillary figure tables
- generic loaders for the ancillary `.dat`, `.txt`, and `.tsv` tables
- minimal physics utilities for spectral functions, Wilson line correlators, cumulants, and rate-equation stepping

## Primary-source basis

The current code basis is anchored to the public artifacts shipped with these papers:

- `2310.18864v1` posted on October 29, 2023:
  `T-matrix analysis of static Wilson line correlators from lattice QCD at finite temperature`
- `2502.09044` posted on February 13, 2025:
  `Quarkonium Spectroscopy in the Quark-Gluon Plasma`
- `2503.10089` posted on March 13, 2025:
  `Non-perturbative quarkonium dissociation rates in strongly coupled quark-gluon plasma`

The 2024 bottomonium paper `2411.09132` and the August 28, 2025 transport paper `2508.20995` are registered as context papers as well, but their arXiv source packages do not expose the same level of ancillary table data.

## Layout

- `src/hefty_tm/papers.py`
  paper metadata and fetch defaults
- `src/hefty_tm/fetch.py`
  source-bundle download and ancillary extraction
- `src/hefty_tm/datasets.py`
  ancillary table loading and summarization
- `src/hefty_tm/spectral.py`
  minimal spectral-function building blocks
- `src/hefty_tm/wilson_line.py`
  Laplace transform, cumulants, and tau-to-zero extrapolation helpers
- `src/hefty_tm/rates.py`
  simple rate-equation stepping utilities
- `tests/test_core.py`
  lightweight regression tests

## Quick start

Install the package and the benchmark dependencies:

```bash
python3 -m pip install -e .
```

List the registered papers:

```bash
PYTHONPATH=src python3 -m hefty_tm list-papers
```

Fetch the default ancillary datasets into `data/external/arxiv/`:

```bash
PYTHONPATH=src python3 -m hefty_tm fetch
```

Summarize a fetched directory:

```bash
PYTHONPATH=src python3 -m hefty_tm summarize-data data/external/arxiv/2310.18864v1
```

Run a toy Wilson-line demo:

```bash
PYTHONPATH=src python3 -m hefty_tm demo-wlc
```

Run the current benchmark branches:

```bash
PYTHONPATH=src python3 -m hefty_tm task1-benchmark --root . --out results/task1
PYTHONPATH=src python3 -m hefty_tm task1-publication-faithful --root . --out results/task1_publication_faithful
PYTHONPATH=src python3 -m hefty_tm task1-publication-locked --root . --out results/task1_publication_locked_potential
PYTHONPATH=src python3 -m hefty_tm task1-publication-smoothed --root . --out results/task1_publication_smoothed_fig4
PYTHONPATH=src python3 -m hefty_tm task1-tang-exact --root . --out results/task1_tang_exact
```

`task1-benchmark` and `task1-publication-faithful` currently call the same
implementation. They differ only in their default output directories.

Run the local tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Scope

This is still not a full reproduction of the HEFTY thermodynamic T-matrix code.
It is intended to:

- collect the public paper artifacts in one place
- provide a clean local interface for inspecting benchmark tables
- encode the core equations needed for the first implementation pass

The full self-consistent thermodynamic T-matrix solver, pole search in the
complex plane, and realistic in-medium kernels still need to be implemented on
top of this scaffold.

See [docs/code_structure.md](docs/code_structure.md) for the current package
layout, branch definitions, and the exact task-1 workflow that is implemented
today.
