from __future__ import annotations

from pathlib import Path

import numpy as np

HBARC = 0.1973269804


def wlc_from_spectrum(
    energies: np.ndarray,
    spectral_density: np.ndarray,
    tau: np.ndarray | float,
) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    spectral_density = np.asarray(spectral_density, dtype=float)
    tau_grid = np.atleast_1d(np.asarray(tau, dtype=float))

    kernel = np.exp(-tau_grid[:, None] * energies[None, :])
    values = np.trapezoid(kernel * spectral_density[None, :], energies, axis=1)
    if np.isscalar(tau):
        return values[0]
    return values


def first_cumulant(tau: np.ndarray, wlc: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    wlc = np.asarray(wlc, dtype=float)
    return -np.gradient(np.log(wlc), tau, edge_order=2)


def tau_zero_extrapolation(
    tau: np.ndarray,
    cumulant: np.ndarray,
    *,
    n_points: int = 5,
) -> float:
    tau = np.asarray(tau, dtype=float)
    cumulant = np.asarray(cumulant, dtype=float)
    if n_points < 2:
        raise ValueError("Need at least two points for tau=0 extrapolation.")
    fit = np.polyfit(tau[:n_points], cumulant[:n_points], deg=1)
    return float(np.polyval(fit, 0.0))


def load_bazavov_wilson_correlator(
    path: str | Path,
    *,
    n_radius: int = 32,
    n_tau: int = 56,
    n_jackknife: int = 16,
) -> np.ndarray:
    data = np.loadtxt(path, dtype=float)
    expected_rows = n_radius * n_tau * (n_jackknife + 1)
    if data.shape[0] != expected_rows:
        raise ValueError(
            f"Unexpected number of rows in {path}: got {data.shape[0]}, expected {expected_rows}."
        )
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least three numeric columns in {path}.")

    samples = data[:, 2].reshape((n_jackknife + 1, n_radius, n_tau)).transpose(1, 2, 0)
    return samples


def bazavov_jackknife_spread(
    central: np.ndarray,
    replicas: np.ndarray,
    *,
    extra_sigma: float = 0.0,
) -> np.ndarray:
    central = np.asarray(central, dtype=float)
    replicas = np.asarray(replicas, dtype=float)
    if replicas.ndim != central.ndim + 1:
        raise ValueError("Replicas must have one extra dimension relative to the central values.")
    delta = central[..., None] - replicas
    return np.sqrt(np.sum(delta**2, axis=-1) + extra_sigma**2)


def bazavov_effective_mass_curve(
    correlator_samples: np.ndarray,
    lattice_spacing_fm: float,
    *,
    tau_extent: int | None = None,
    additive_offset_gev: float = 0.0,
    additive_offset_sigma_gev: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(correlator_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("Expected correlator samples with shape (n_tau, n_samples).")
    if samples.shape[1] < 2:
        raise ValueError("Need one central sample plus at least one jackknife replica.")

    n_tau_total = samples.shape[0]
    n_tau_used = n_tau_total if tau_extent is None else int(min(tau_extent, n_tau_total))
    if n_tau_used < 2:
        raise ValueError("Need at least two Euclidean-time points to form an effective mass.")

    trimmed = samples[:n_tau_used]
    central = trimmed[:, 0]
    replicas = trimmed[:, 1:]

    prefactor = -HBARC / float(lattice_spacing_fm)
    meff = prefactor * np.log(np.abs(central[1:] / central[:-1])) + additive_offset_gev
    replica_meff = prefactor * np.log(np.abs(replicas[1:] / replicas[:-1])) + additive_offset_gev
    sigma = bazavov_jackknife_spread(meff, replica_meff, extra_sigma=additive_offset_sigma_gev)
    tau_fm = float(lattice_spacing_fm) * np.arange(1, n_tau_used, dtype=float)
    return tau_fm, meff, sigma
