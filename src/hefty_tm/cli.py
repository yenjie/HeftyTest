from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .datasets import summarize_tables
from .fetch import fetch_many
from .papers import DEFAULT_FETCH_KEYS, PAPER_REGISTRY
from .spectral import breit_wigner_spectral_function
from .wilson_line import first_cumulant, tau_zero_extrapolation, wlc_from_spectrum


def _import_task1_runners():
    from .benchmark_task1 import (
        run_task1_benchmark,
        run_task1_publication_locked_benchmark,
        run_task1_publication_smoothed_benchmark,
        run_task1_tang_exact_benchmark,
        run_task1_tang_inferred_medium_benchmark,
    )

    return (
        run_task1_benchmark,
        run_task1_publication_locked_benchmark,
        run_task1_publication_smoothed_benchmark,
        run_task1_tang_exact_benchmark,
        run_task1_tang_inferred_medium_benchmark,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hefty_tm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-papers")

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument(
        "keys",
        nargs="*",
        default=list(DEFAULT_FETCH_KEYS),
        help="Paper keys to fetch. Defaults to the papers with ancillary tables.",
    )
    fetch_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/external/arxiv"),
        help="Destination directory for extracted source assets.",
    )
    fetch_parser.add_argument(
        "--include-full-source",
        action="store_true",
        help="Keep top-level TeX and PDF files in addition to ancillary tables.",
    )

    summary_parser = subparsers.add_parser("summarize-data")
    summary_parser.add_argument("root", type=Path)

    task1_parser = subparsers.add_parser("task1-benchmark")
    task1_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing lattice and ancillary data.",
    )
    task1_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/task1"),
        help="Destination directory for plots, fit parameters, and report.",
    )

    task1_faithful_parser = subparsers.add_parser("task1-publication-faithful")
    task1_faithful_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing lattice and ancillary data.",
    )
    task1_faithful_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/task1_publication_faithful"),
        help="Destination directory for the publication-faithful reduced-static benchmark outputs.",
    )

    task1_locked_parser = subparsers.add_parser("task1-publication-locked")
    task1_locked_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing lattice and ancillary data.",
    )
    task1_locked_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/task1_publication_locked_potential"),
        help="Destination directory for the figure-locked publication-surrogate benchmark outputs.",
    )

    task1_smoothed_parser = subparsers.add_parser("task1-publication-smoothed")
    task1_smoothed_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing lattice and ancillary data.",
    )
    task1_smoothed_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/task1_publication_smoothed_fig4"),
        help="Destination directory for the publication-smoothed Figure 4 benchmark outputs.",
    )

    task1_tang_exact_parser = subparsers.add_parser("task1-tang-exact")
    task1_tang_exact_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing lattice and ancillary data.",
    )
    task1_tang_exact_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/task1_tang_exact"),
        help="Destination directory for the Tang-style reduced static replay outputs.",
    )

    task1_tang_inferred_medium_parser = subparsers.add_parser("task1-tang-inferred-medium")
    task1_tang_inferred_medium_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing lattice and ancillary data.",
    )
    task1_tang_inferred_medium_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/task1_tang_inferred_medium"),
        help="Destination directory for the Tang replay with inferred missing-medium corrections.",
    )

    subparsers.add_parser("demo-wlc")
    return parser


def _list_papers() -> int:
    for key, paper in PAPER_REGISTRY.items():
        ancillary = "yes" if paper.has_ancillary_tables else "no"
        print(f"{key}: {paper.arxiv_id} | posted {paper.posted} | ancillary={ancillary}")
        print(f"  {paper.title}")
        print(f"  role: {paper.role}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    results = fetch_many(
        args.keys,
        args.out,
        include_full_source=args.include_full_source,
    )
    print(json.dumps(results, indent=2))
    return 0


def _summarize_data(args: argparse.Namespace) -> int:
    for item in summarize_tables(args.root):
        print(json.dumps(item, sort_keys=True))
    return 0


def _demo_wlc() -> int:
    energies = np.linspace(-1.0, 4.0, 2000)
    rho = breit_wigner_spectral_function(energies, resonance_energy=0.8, width=0.25)
    tau = np.linspace(0.01, 2.00, 60)
    wlc = wlc_from_spectrum(energies, rho, tau)
    m1 = first_cumulant(tau, wlc)
    tau0 = tau_zero_extrapolation(tau, m1)
    print(json.dumps({"tau0_extrapolated_cumulant": tau0, "points": int(tau.size)}, indent=2))
    return 0


def _task1_benchmark(args: argparse.Namespace) -> int:
    run_task1_benchmark, _, _, _, _ = _import_task1_runners()
    payload = run_task1_benchmark(root=args.root.resolve(), output_dir=args.out.resolve())
    print(json.dumps(payload["global_common_ms_fit"], indent=2))
    return 0


def _task1_publication_faithful(args: argparse.Namespace) -> int:
    run_task1_benchmark, _, _, _, _ = _import_task1_runners()
    payload = run_task1_benchmark(root=args.root.resolve(), output_dir=args.out.resolve())
    print(json.dumps(payload["global_common_ms_fit"], indent=2))
    return 0


def _task1_publication_locked(args: argparse.Namespace) -> int:
    _, run_task1_publication_locked_benchmark, _, _, _ = _import_task1_runners()
    payload = run_task1_publication_locked_benchmark(
        root=args.root.resolve(),
        output_dir=args.out.resolve(),
    )
    print(json.dumps(payload["global_common_ms_fit"], indent=2))
    return 0


def _task1_publication_smoothed(args: argparse.Namespace) -> int:
    _, _, run_task1_publication_smoothed_benchmark, _, _ = _import_task1_runners()
    payload = run_task1_publication_smoothed_benchmark(
        root=args.root.resolve(),
        output_dir=args.out.resolve(),
    )
    print(json.dumps(payload["global_common_ms_fit"], indent=2))
    return 0


def _task1_tang_exact(args: argparse.Namespace) -> int:
    _, _, _, run_task1_tang_exact_benchmark, _ = _import_task1_runners()
    payload = run_task1_tang_exact_benchmark(
        root=args.root.resolve(),
        output_dir=args.out.resolve(),
    )
    print(json.dumps(payload["global_common_ms_fit"], indent=2))
    return 0


def _task1_tang_inferred_medium(args: argparse.Namespace) -> int:
    _, _, _, _, run_task1_tang_inferred_medium_benchmark = _import_task1_runners()
    payload = run_task1_tang_inferred_medium_benchmark(
        root=args.root.resolve(),
        output_dir=args.out.resolve(),
    )
    print(json.dumps(payload["global_common_ms_fit"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-papers":
        return _list_papers()
    if args.command == "fetch":
        return _fetch(args)
    if args.command == "summarize-data":
        return _summarize_data(args)
    if args.command == "task1-benchmark":
        return _task1_benchmark(args)
    if args.command == "task1-publication-faithful":
        return _task1_publication_faithful(args)
    if args.command == "task1-publication-locked":
        return _task1_publication_locked(args)
    if args.command == "task1-publication-smoothed":
        return _task1_publication_smoothed(args)
    if args.command == "task1-tang-exact":
        return _task1_tang_exact(args)
    if args.command == "task1-tang-inferred-medium":
        return _task1_tang_inferred_medium(args)
    if args.command == "demo-wlc":
        return _demo_wlc()
    parser.error(f"Unknown command: {args.command}")
    return 2
