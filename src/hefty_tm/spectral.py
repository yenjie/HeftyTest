from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabulatedRadialProfile:
    radius: np.ndarray
    values: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        radius = np.asarray(self.radius, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if radius.ndim != 1 or values.ndim != 1:
            raise ValueError("Tabulated profiles must be one-dimensional.")
        if radius.size != values.size:
            raise ValueError("Radius and value arrays must have matching lengths.")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "values", values)

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        return np.interp(x, self.radius, self.values)


def constant_self_energy(
    energies: np.ndarray | float,
    *,
    mass_shift: float = 0.0,
    width: float = 0.0,
) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    return np.full_like(energies, mass_shift - 0.5j * width, dtype=complex)


def qqbar_spectral_function(
    energies: np.ndarray,
    potential: np.ndarray | float,
    self_energy: np.ndarray | complex,
    *,
    phi: np.ndarray | float = 1.0,
) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    potential = np.asarray(potential, dtype=float)
    self_energy = np.asarray(self_energy, dtype=complex)
    phi = np.asarray(phi, dtype=float)

    denominator = energies - potential - phi * self_energy
    propagator = 1.0 / denominator
    return -np.imag(propagator) / np.pi


def breit_wigner_spectral_function(
    energies: np.ndarray,
    resonance_energy: float,
    width: float,
) -> np.ndarray:
    sigma = constant_self_energy(energies, width=width)
    return qqbar_spectral_function(energies, resonance_energy, sigma)


def pole_width_from_imaginary_energy(imaginary_part: float) -> float:
    return -2.0 * imaginary_part
