from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    key: str
    arxiv_id: str
    posted: str
    title: str
    role: str
    source_url: str
    has_ancillary_tables: bool
    note: str = ""


PAPER_REGISTRY: dict[str, Paper] = {
    "wlc-2024": Paper(
        key="wlc-2024",
        arxiv_id="2310.18864v1",
        posted="2023-10-29",
        title="T-matrix analysis of static Wilson line correlators from lattice QCD at finite temperature",
        role="Core extraction paper for Wilson line correlators, interference function, spectral functions, and diffusion benchmarks.",
        source_url="https://arxiv.org/e-print/2310.18864v1",
        has_ancillary_tables=True,
        note="The journal article states that the datasets are available in the ancillary files accompanying the arXiv version.",
    ),
    "bottomonium-2024": Paper(
        key="bottomonium-2024",
        arxiv_id="2411.09132",
        posted="2024-11-14",
        title="Bottomonium Properties in QGP from a Lattice-QCD Informed T-Matrix Approach",
        role="Follow-up extraction paper using extended-operator bottomonium correlators.",
        source_url="https://arxiv.org/e-print/2411.09132",
        has_ancillary_tables=False,
        note="Public arXiv source package appears to contain manuscript sources and figure PDFs, but no ancillary table directory.",
    ),
    "spectroscopy-2025": Paper(
        key="spectroscopy-2025",
        arxiv_id="2502.09044",
        posted="2025-02-13",
        title="Quarkonium Spectroscopy in the Quark-Gluon Plasma",
        role="Complex-energy pole analysis and melting criterion.",
        source_url="https://arxiv.org/e-print/2502.09044",
        has_ancillary_tables=True,
        note="Source package includes an anc/ directory with figure tables.",
    ),
    "rates-2025": Paper(
        key="rates-2025",
        arxiv_id="2503.10089",
        posted="2025-03-13",
        title="Non-perturbative quarkonium dissociation rates in strongly coupled quark-gluon plasma",
        role="Reaction-rate paper built on the same lattice-constrained kernel.",
        source_url="https://arxiv.org/e-print/2503.10089",
        has_ancillary_tables=True,
        note="Source package includes an anc/ directory with extensive rate and benchmark tables.",
    ),
    "transport-2025": Paper(
        key="transport-2025",
        arxiv_id="2508.20995",
        posted="2025-08-28",
        title="Bottomonium transport in a strongly coupled quark-gluon plasma",
        role="Latest transport application that consumes lattice-constrained reaction rates.",
        source_url="https://arxiv.org/e-print/2508.20995",
        has_ancillary_tables=False,
        note="Public arXiv source package appears to contain manuscript sources and figure PDFs, but no ancillary table directory.",
    ),
}

DEFAULT_FETCH_KEYS: tuple[str, ...] = tuple(
    paper.key for paper in PAPER_REGISTRY.values() if paper.has_ancillary_tables
)


def get_paper(key: str) -> Paper:
    try:
        return PAPER_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(PAPER_REGISTRY))
        raise KeyError(f"Unknown paper key '{key}'. Known keys: {known}") from exc
