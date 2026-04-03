from __future__ import annotations

import numpy as np


def thermal_boltzmann_weight(energy: np.ndarray | float, temperature: float) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    return np.exp(-energy / temperature)


def weighted_spectral_rate(
    energies: np.ndarray,
    spectral_density: np.ndarray,
    temperature: float,
) -> float:
    energies = np.asarray(energies, dtype=float)
    spectral_density = np.asarray(spectral_density, dtype=float)
    weights = thermal_boltzmann_weight(np.clip(energies, 0.0, None), temperature)
    return float(np.trapezoid(weights * spectral_density, energies))


def rate_equation_step(
    occupancy: float,
    *,
    gain: float,
    loss_rate: float,
    dt: float,
) -> float:
    return occupancy + dt * (gain - loss_rate * occupancy)


def solve_rate_equation(
    times: np.ndarray,
    initial_occupancy: float,
    *,
    gain: np.ndarray | float,
    loss_rate: np.ndarray | float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    gain_array = np.broadcast_to(np.asarray(gain, dtype=float), times.shape)
    loss_array = np.broadcast_to(np.asarray(loss_rate, dtype=float), times.shape)

    occupancy = np.empty_like(times, dtype=float)
    occupancy[0] = initial_occupancy
    for idx in range(1, times.size):
        dt = times[idx] - times[idx - 1]
        occupancy[idx] = rate_equation_step(
            occupancy[idx - 1],
            gain=float(gain_array[idx - 1]),
            loss_rate=float(loss_array[idx - 1]),
            dt=float(dt),
        )
    return occupancy
