from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


DISTANCE_BENCHMARK_FM = (0.224, 0.505, 0.757)
TEMPERATURE_BENCHMARK_GEV = (0.195, 0.251, 0.293, 0.352)
DEFAULT_STATIC_ENERGY_MIN = -1.0
DEFAULT_STATIC_ENERGY_MAX = 3.0
DEFAULT_STATIC_ENERGY_POINTS = 601


@dataclass(frozen=True)
class SelfEnergyKernel:
    temperature_gev: float
    energies: np.ndarray
    real_part: np.ndarray
    imag_part: np.ndarray

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=float)
        real_part = np.asarray(self.real_part, dtype=float)
        imag_part = np.asarray(self.imag_part, dtype=float)
        if energies.ndim != 1:
            raise ValueError("Self-energy energies must be one-dimensional.")
        if real_part.shape != energies.shape or imag_part.shape != energies.shape:
            raise ValueError("Self-energy arrays must match the energy grid.")
        if np.any(imag_part >= 0.0):
            raise ValueError("Imaginary self-energy must stay negative.")
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "real_part", real_part)
        object.__setattr__(self, "imag_part", imag_part)


@dataclass(frozen=True)
class PublicOuterLoopAnchor:
    temperature_gev: float
    delta_re0: float
    sigma_re0: float
    delta_re1: float
    sigma_re1: float
    delta_re2: float
    sigma_re2: float
    delta_im_log0: float
    sigma_im_log0: float
    delta_im1: float
    sigma_im1: float
    delta_im2: float
    sigma_im2: float


def energy_shape_coordinate(energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    half_span = 0.5 * (float(energies.max()) - float(energies.min()))
    if half_span <= 0.0:
        raise ValueError("Energy grid must span a finite interval.")
    return (energies - float(np.mean(energies))) / half_span


def _temperature_column_index(temperature_gev: float) -> int:
    mapping = {0.195: 1, 0.251: 2, 0.293: 3, 0.352: 4}
    return mapping[round(float(temperature_gev), 3)]


def load_tang_phi_table(root: Path) -> np.ndarray:
    path = root / "data" / "external" / "arxiv" / "2310.18864v1" / "anc" / "Fig3.dat"
    return np.loadtxt(path)


def load_tang_fig5(root: Path) -> np.ndarray:
    path = root / "data" / "external" / "arxiv" / "2310.18864v1" / "anc" / "Fig5.dat"
    return np.loadtxt(path)


def load_tang_fig6(root: Path) -> np.ndarray:
    path = root / "data" / "external" / "arxiv" / "2310.18864v1" / "anc" / "Fig6.dat"
    return np.loadtxt(path)


def default_static_energy_grid(
    *,
    energy_min: float = DEFAULT_STATIC_ENERGY_MIN,
    energy_max: float = DEFAULT_STATIC_ENERGY_MAX,
    n_points: int = DEFAULT_STATIC_ENERGY_POINTS,
) -> np.ndarray:
    return np.linspace(float(energy_min), float(energy_max), int(n_points))


def build_phi_interpolators(root: Path) -> dict[float, PchipInterpolator]:
    table = load_tang_phi_table(root)
    return {
        temperature_gev: PchipInterpolator(
            table[:, 0],
            table[:, _temperature_column_index(temperature_gev)],
            extrapolate=True,
        )
        for temperature_gev in TEMPERATURE_BENCHMARK_GEV
    }


def _infer_self_energy_at_energy(
    energy: float,
    rho_values: np.ndarray,
    potential_values: np.ndarray,
    phi_values: np.ndarray,
) -> tuple[float, float]:
    def residuals(params: np.ndarray) -> np.ndarray:
        real_sigma, log_width = params
        imag_sigma = -np.exp(log_width)
        real_denominator = energy - potential_values - phi_values * real_sigma
        width = -phi_values * imag_sigma
        predicted = width / (np.pi * (real_denominator**2 + width**2))
        scale = np.maximum(rho_values, 2.0e-3)
        return (predicted - rho_values) / scale

    result = least_squares(
        residuals,
        x0=np.array([0.0, np.log(0.1)], dtype=float),
        bounds=([-5.0, -10.0], [5.0, 2.0]),
    )
    return float(result.x[0]), float(-np.exp(result.x[1]))


def infer_reference_self_energy_kernels(
    root: Path,
    *,
    smoothing_window: int | dict[float, int] = 7,
    imag_smoothing_window: int | dict[float, int] | None = None,
    dense_energy_points: int = 1201,
) -> dict[float, SelfEnergyKernel]:
    phi_interpolators = build_phi_interpolators(root)
    fig5 = load_tang_fig5(root)
    fig6 = load_tang_fig6(root)

    potential_interpolators = {
        temperature_gev: PchipInterpolator(
            fig5[:, 0],
            fig5[:, _temperature_column_index(temperature_gev)],
            extrapolate=True,
        )
        for temperature_gev in TEMPERATURE_BENCHMARK_GEV
    }

    column_map: dict[tuple[float, float], int] = {}
    col_idx = 1
    for distance_fm in DISTANCE_BENCHMARK_FM:
        for temperature_gev in TEMPERATURE_BENCHMARK_GEV:
            column_map[(distance_fm, temperature_gev)] = col_idx
            col_idx += 1

    energy_grid = fig6[:, 0]
    kernels: dict[float, SelfEnergyKernel] = {}

    for temperature_gev in TEMPERATURE_BENCHMARK_GEV:
        if isinstance(smoothing_window, dict):
            window = int(smoothing_window[round(float(temperature_gev), 3)])
        else:
            window = int(smoothing_window)
        if imag_smoothing_window is None:
            imag_window = window
        elif isinstance(imag_smoothing_window, dict):
            imag_window = int(imag_smoothing_window[round(float(temperature_gev), 3)])
        else:
            imag_window = int(imag_smoothing_window)
        potential_values = np.array(
            [float(potential_interpolators[temperature_gev](r)) for r in DISTANCE_BENCHMARK_FM],
            dtype=float,
        )
        phi_values = np.array(
            [float(phi_interpolators[temperature_gev](r)) for r in DISTANCE_BENCHMARK_FM],
            dtype=float,
        )
        real_sigma = []
        imag_sigma = []
        for row_idx, energy in enumerate(energy_grid):
            rho_values = np.array(
                [fig6[row_idx, column_map[(distance_fm, temperature_gev)]] for distance_fm in DISTANCE_BENCHMARK_FM],
                dtype=float,
            )
            real_part, imag_part = _infer_self_energy_at_energy(
                float(energy),
                rho_values,
                potential_values,
                phi_values,
            )
            real_sigma.append(real_part)
            imag_sigma.append(imag_part)

        real_sigma = savgol_filter(np.asarray(real_sigma, dtype=float), window, 3, mode="interp")
        imag_sigma = -np.exp(
            savgol_filter(np.log(-np.asarray(imag_sigma, dtype=float)), imag_window, 3, mode="interp")
        )

        dense_energy_grid = np.linspace(float(energy_grid.min()), float(energy_grid.max()), dense_energy_points)
        kernels[temperature_gev] = SelfEnergyKernel(
            temperature_gev=temperature_gev,
            energies=dense_energy_grid,
            real_part=PchipInterpolator(energy_grid, real_sigma, extrapolate=True)(dense_energy_grid),
            imag_part=PchipInterpolator(energy_grid, imag_sigma, extrapolate=True)(dense_energy_grid),
        )
    return kernels


def polynomial_self_energy_kernel(
    *,
    temperature_gev: float,
    energies: np.ndarray | None = None,
    re_constant: float = 0.0,
    re_slope: float = 0.0,
    re_curvature: float = 0.0,
    im_log_amplitude: float = -2.0,
    im_slope: float = 0.0,
    im_curvature: float = 0.0,
) -> SelfEnergyKernel:
    energy_grid = default_static_energy_grid() if energies is None else np.asarray(energies, dtype=float)
    x = energy_shape_coordinate(energy_grid)
    x2 = x**2 - (1.0 / 3.0)
    real_part = re_constant + re_slope * x + re_curvature * x2
    imag_part = -np.exp(im_log_amplitude + im_slope * x + im_curvature * x2)
    return SelfEnergyKernel(
        temperature_gev=float(temperature_gev),
        energies=energy_grid,
        real_part=real_part,
        imag_part=imag_part,
    )


def fit_absolute_polynomial_kernel_parameters(
    kernel: SelfEnergyKernel,
) -> tuple[float, float, float, float, float, float]:
    energies = np.asarray(kernel.energies, dtype=float)
    x = energy_shape_coordinate(energies)
    x2 = x**2 - (1.0 / 3.0)
    design = np.column_stack([np.ones_like(x), x, x2])
    real_coeffs, *_ = np.linalg.lstsq(design, np.asarray(kernel.real_part, dtype=float), rcond=None)
    imag_coeffs, *_ = np.linalg.lstsq(design, np.log(np.clip(-np.asarray(kernel.imag_part, dtype=float), 1.0e-16, None)), rcond=None)
    return (
        float(real_coeffs[0]),
        float(real_coeffs[1]),
        float(real_coeffs[2]),
        float(imag_coeffs[0]),
        float(imag_coeffs[1]),
        float(imag_coeffs[2]),
    )


def _fit_polynomial_shape_coefficients(
    energies: np.ndarray,
    real_part: np.ndarray,
    imag_part: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    energies = np.asarray(energies, dtype=float)
    real_part = np.asarray(real_part, dtype=float)
    imag_part = np.asarray(imag_part, dtype=float)
    x = energy_shape_coordinate(energies)
    x2 = x**2 - (1.0 / 3.0)
    design = np.column_stack([np.ones_like(x), x, x2])

    real_scale = max(float(np.std(real_part)), 1.0e-6)
    real_shape = (real_part - float(np.mean(real_part))) / real_scale
    real_coeffs, *_ = np.linalg.lstsq(design, real_shape, rcond=None)

    imag_peak = max(float(np.max(imag_part)), 1.0e-12)
    imag_shape = np.log(np.clip(imag_part / imag_peak, 1.0e-12, None))
    imag_coeffs, *_ = np.linalg.lstsq(design, imag_shape, rcond=None)
    imag_log_peak = float(np.log(imag_peak))
    return np.asarray(real_coeffs, dtype=float), imag_log_peak, np.asarray(imag_coeffs, dtype=float)


def load_public_outer_loop_anchors(
    root: Path,
    *,
    energy_min: float = 0.0,
    energy_max: float = 3.0,
    n_energy: int = 301,
) -> dict[float, PublicOuterLoopAnchor]:
    anc = root / "data" / "external" / "arxiv" / "2503.10089" / "anc"
    benchmark_pairs = {
        0.195: (195, 194),
        0.251: (251, 258),
        0.293: (293, 320),
        0.352: (352, 400),
    }
    energies = np.linspace(float(energy_min), float(energy_max), int(n_energy))

    wlc_b = np.loadtxt(anc / "Fig6_WLC_b.txt")
    scs_b = np.loadtxt(anc / "Fig6_SCS_b.txt")
    wlc_b_interp = PchipInterpolator(wlc_b[:, 0], wlc_b[:, 1], extrapolate=True)
    scs_b_interp = PchipInterpolator(scs_b[:, 0], scs_b[:, 1], extrapolate=True)

    anchors: dict[float, PublicOuterLoopAnchor] = {}
    for temperature_gev, (wlc_tag, scs_tag) in benchmark_pairs.items():
        wlc_cq_real = np.loadtxt(anc / f"Fig4_WLC_cq_real_T{wlc_tag}.txt")
        wlc_cq_imag = np.loadtxt(anc / f"Fig4_WLC_cq_imag_T{wlc_tag}.txt")
        wlc_cg_real = np.loadtxt(anc / f"Fig4_WLC_cg_real_T{wlc_tag}.txt")
        wlc_cg_imag = np.loadtxt(anc / f"Fig4_WLC_cg_imag_T{wlc_tag}.txt")
        scs_cq_real = np.loadtxt(anc / f"Fig4_SCS_cq_real_T{scs_tag}.txt")
        scs_cq_imag = np.loadtxt(anc / f"Fig4_SCS_cq_imag_T{scs_tag}.txt")
        scs_cg_real = np.loadtxt(anc / f"Fig4_SCS_cg_real_T{scs_tag}.txt")
        scs_cg_imag = np.loadtxt(anc / f"Fig4_SCS_cg_imag_T{scs_tag}.txt")

        wlc_real = PchipInterpolator(wlc_cq_real[:, 0], wlc_cq_real[:, 1], extrapolate=True)(energies)
        wlc_real += PchipInterpolator(wlc_cg_real[:, 0], wlc_cg_real[:, 1], extrapolate=True)(energies)
        scs_real = PchipInterpolator(scs_cq_real[:, 0], scs_cq_real[:, 1], extrapolate=True)(energies)
        scs_real += PchipInterpolator(scs_cg_real[:, 0], scs_cg_real[:, 1], extrapolate=True)(energies)

        wlc_imag = PchipInterpolator(wlc_cq_imag[:, 0], wlc_cq_imag[:, 1], extrapolate=True)(energies)
        wlc_imag += PchipInterpolator(wlc_cg_imag[:, 0], wlc_cg_imag[:, 1], extrapolate=True)(energies)
        scs_imag = PchipInterpolator(scs_cq_imag[:, 0], scs_cq_imag[:, 1], extrapolate=True)(energies)
        scs_imag += PchipInterpolator(scs_cg_imag[:, 0], scs_cg_imag[:, 1], extrapolate=True)(energies)

        wlc_real_coeffs, wlc_imag_log_peak, wlc_imag_coeffs = _fit_polynomial_shape_coefficients(
            energies, wlc_real, wlc_imag
        )
        scs_real_coeffs, scs_imag_log_peak, scs_imag_coeffs = _fit_polynomial_shape_coefficients(
            energies, scs_real, scs_imag
        )

        delta_re = scs_real_coeffs - wlc_real_coeffs
        delta_im = scs_imag_coeffs - wlc_imag_coeffs
        delta_im_log0 = scs_imag_log_peak - wlc_imag_log_peak
        temperature_mev = temperature_gev * 1000.0
        delta_re0 = float(scs_b_interp(temperature_mev) - wlc_b_interp(temperature_mev))

        anchors[temperature_gev] = PublicOuterLoopAnchor(
            temperature_gev=temperature_gev,
            delta_re0=delta_re0,
            sigma_re0=max(abs(delta_re0), 0.06),
            delta_re1=float(delta_re[1]),
            sigma_re1=max(abs(float(delta_re[1])), 0.20),
            delta_re2=float(delta_re[2]),
            sigma_re2=max(abs(float(delta_re[2])), 0.20),
            delta_im_log0=float(delta_im_log0),
            sigma_im_log0=max(abs(float(delta_im_log0)), 0.35),
            delta_im1=float(delta_im[1]),
            sigma_im1=max(abs(float(delta_im[1])), 0.35),
            delta_im2=float(delta_im[2]),
            sigma_im2=max(abs(float(delta_im[2])), 0.35),
        )
    return anchors


def static_spectral_function(
    *,
    potential: float,
    phi_value: float,
    kernel: SelfEnergyKernel,
    re_sigma_offset: float = 0.0,
    re_sigma_scale: float = 1.0,
    re_sigma_slope: float = 0.0,
    re_sigma_curvature: float = 0.0,
    re_sigma_radius: float = 0.0,
    re_sigma_radius_curvature: float = 0.0,
    re_sigma_radius_mid: float = 0.0,
    im_sigma_scale: float = 1.0,
    im_sigma_slope: float = 0.0,
    im_sigma_curvature: float = 0.0,
    im_sigma_radius: float = 0.0,
    im_sigma_radius_curvature: float = 0.0,
    im_sigma_radius_mid: float = 0.0,
    im_sigma_bias: float = 0.0,
    spectral_shoulder_amp: float = 0.0,
    spectral_shoulder_offset: float = 0.0,
    spectral_shoulder_width: float = 0.0,
    spectral_soft_mode_amp: float = 0.0,
    spectral_soft_mode_drop: float = 0.0,
    spectral_soft_mode_width: float = 0.0,
    gluon_gap_strength: float = 0.0,
    gluon_gap_gev: float = 0.0,
    distance_shape: float = 0.0,
) -> np.ndarray:
    shape_coordinate = energy_shape_coordinate(kernel.energies)
    shape_curvature = shape_coordinate**2 - (1.0 / 3.0)
    distance_curvature = distance_shape * (distance_shape - 0.5)
    distance_mid = distance_shape * (1.0 - distance_shape)
    real_part = (
        re_sigma_offset
        + re_sigma_scale * kernel.real_part
        + re_sigma_slope * shape_coordinate
        + re_sigma_curvature * shape_curvature
        + re_sigma_radius * distance_shape
        + re_sigma_radius_curvature * distance_curvature
        + re_sigma_radius_mid * distance_mid
    )
    imag_part = kernel.imag_part * im_sigma_scale * np.exp(
        im_sigma_slope * shape_coordinate
        + im_sigma_curvature * shape_curvature
        + im_sigma_radius * distance_shape
        + im_sigma_radius_curvature * distance_curvature
        + im_sigma_radius_mid * distance_mid
    ) - im_sigma_bias * distance_shape
    gap_strength = max(float(gluon_gap_strength), 0.0)
    gap_gev = max(float(gluon_gap_gev), 0.0)
    if gap_strength > 0.0 and gap_gev > 0.0:
        soft_distance = max(float(distance_shape), 0.0)
        gap_center = float(potential - 0.6 * gap_gev * soft_distance * soft_distance)
        gap_width = gap_gev * (0.90 + 0.35 * soft_distance)
        gap_profile = np.exp(-0.5 * ((kernel.energies - gap_center) / gap_width) ** 2)
        damping = np.clip(1.0 - gap_strength * gap_profile, 0.15, 1.0)
        imag_part = imag_part * damping
    real_denominator = kernel.energies - potential - phi_value * real_part
    width = -phi_value * imag_part
    spectral_density = width / (np.pi * (real_denominator**2 + width**2))
    shoulder_amp = max(float(spectral_shoulder_amp), 0.0)
    shoulder_width = max(float(spectral_shoulder_width), 0.0)
    if shoulder_amp > 0.0 and shoulder_width > 0.0:
        shoulder_center = float(potential + spectral_shoulder_offset)
        shoulder_profile = np.exp(-0.5 * ((kernel.energies - shoulder_center) / shoulder_width) ** 2)
        shoulder_profile /= np.sqrt(2.0 * np.pi) * shoulder_width
        transfer_fraction = np.clip(shoulder_amp * max(distance_shape, 0.0), 0.0, 0.85)
        spectral_weight = float(np.trapezoid(spectral_density, kernel.energies))
        spectral_density = (
            (1.0 - transfer_fraction) * spectral_density
            + transfer_fraction * spectral_weight * shoulder_profile
        )
    soft_mode_amp = max(float(spectral_soft_mode_amp), 0.0)
    soft_mode_width = max(float(spectral_soft_mode_width), 0.0)
    if soft_mode_amp > 0.0 and soft_mode_width > 0.0:
        soft_distance = max(float(distance_shape), 0.0)
        soft_center = float(potential - spectral_soft_mode_drop * soft_distance * soft_distance)
        effective_width = soft_mode_width * (0.70 + 0.45 * soft_distance)
        soft_profile = np.exp(-0.5 * ((kernel.energies - soft_center) / effective_width) ** 2)
        soft_profile /= np.sqrt(2.0 * np.pi) * effective_width
        transfer_fraction = np.clip(soft_mode_amp * (0.15 + 1.00 * soft_distance), 0.0, 0.75)
        spectral_weight = float(np.trapezoid(spectral_density, kernel.energies))
        spectral_density = (
            (1.0 - transfer_fraction) * spectral_density
            + transfer_fraction * spectral_weight * soft_profile
        )
    return spectral_density


def model_cumulant_curve(
    *,
    temperature_gev: float,
    tau_t: np.ndarray,
    potential: float,
    phi_value: float,
    kernel: SelfEnergyKernel,
    re_sigma_offset: float = 0.0,
    re_sigma_scale: float = 1.0,
    re_sigma_slope: float = 0.0,
    re_sigma_curvature: float = 0.0,
    re_sigma_radius: float = 0.0,
    re_sigma_radius_curvature: float = 0.0,
    re_sigma_radius_mid: float = 0.0,
    im_sigma_scale: float = 1.0,
    im_sigma_slope: float = 0.0,
    im_sigma_curvature: float = 0.0,
    im_sigma_radius: float = 0.0,
    im_sigma_radius_curvature: float = 0.0,
    im_sigma_radius_mid: float = 0.0,
    im_sigma_bias: float = 0.0,
    spectral_shoulder_amp: float = 0.0,
    spectral_shoulder_offset: float = 0.0,
    spectral_shoulder_width: float = 0.0,
    spectral_soft_mode_amp: float = 0.0,
    spectral_soft_mode_drop: float = 0.0,
    spectral_soft_mode_width: float = 0.0,
    gluon_gap_strength: float = 0.0,
    gluon_gap_gev: float = 0.0,
    distance_shape: float = 0.0,
    anchor_to_potential: bool = True,
) -> np.ndarray:
    tau_t = np.asarray(tau_t, dtype=float)
    spectral_density = static_spectral_function(
        potential=potential,
        phi_value=phi_value,
        kernel=kernel,
        re_sigma_offset=re_sigma_offset,
        re_sigma_scale=re_sigma_scale,
        re_sigma_slope=re_sigma_slope,
        re_sigma_curvature=re_sigma_curvature,
        re_sigma_radius=re_sigma_radius,
        re_sigma_radius_curvature=re_sigma_radius_curvature,
        re_sigma_radius_mid=re_sigma_radius_mid,
        im_sigma_scale=im_sigma_scale,
        im_sigma_slope=im_sigma_slope,
        im_sigma_curvature=im_sigma_curvature,
        im_sigma_radius=im_sigma_radius,
        im_sigma_radius_curvature=im_sigma_radius_curvature,
        im_sigma_radius_mid=im_sigma_radius_mid,
        im_sigma_bias=im_sigma_bias,
        spectral_shoulder_amp=spectral_shoulder_amp,
        spectral_shoulder_offset=spectral_shoulder_offset,
        spectral_shoulder_width=spectral_shoulder_width,
        spectral_soft_mode_amp=spectral_soft_mode_amp,
        spectral_soft_mode_drop=spectral_soft_mode_drop,
        spectral_soft_mode_width=spectral_soft_mode_width,
        gluon_gap_strength=gluon_gap_strength,
        gluon_gap_gev=gluon_gap_gev,
        distance_shape=distance_shape,
    )
    tau_phys = tau_t / float(temperature_gev)
    kernel_matrix = np.exp(-tau_phys[:, None] * kernel.energies[None, :])
    weights = spectral_density[None, :] * kernel_matrix
    w0 = np.trapezoid(weights, kernel.energies, axis=1)
    w1 = np.trapezoid(weights * kernel.energies[None, :], kernel.energies, axis=1)
    raw_cumulant = w1 / np.clip(w0, 1.0e-16, None)
    if not anchor_to_potential:
        return raw_cumulant
    raw_tau0 = float(
        np.trapezoid(kernel.energies * spectral_density, kernel.energies)
        / np.clip(np.trapezoid(spectral_density, kernel.energies), 1.0e-16, None)
    )
    return potential + (raw_cumulant - raw_tau0)
