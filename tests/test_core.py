from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hefty_tm.benchmark_task1 import (
    _fit_peak_ansatz,
    _mix_bazavov_profiles,
    _phi_from_fit,
    fit_global_common_ms,
    load_tang_fig4_targets,
    load_tang_fig5_targets,
    PotentialFit,
    screened_cornell_potential,
    summarize_spectral_curve,
    weighted_quadratic_extrapolation,
)
from hefty_tm.datasets import load_table
from hefty_tm.fetch import safe_extract_member_path
from hefty_tm.rates import solve_rate_equation
from hefty_tm.spectral import TabulatedRadialProfile, breit_wigner_spectral_function
from hefty_tm.static_tmatrix import SelfEnergyKernel, load_public_outer_loop_anchors, model_cumulant_curve
from hefty_tm.static_tmatrix import fit_absolute_polynomial_kernel_parameters, polynomial_self_energy_kernel
from hefty_tm.wilson_line import (
    bazavov_effective_mass_curve,
    first_cumulant,
    load_bazavov_wilson_correlator,
    wlc_from_spectrum,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_exponential_cumulant_is_constant(self) -> None:
        tau = np.linspace(0.0, 1.0, 101)
        mass = 1.7
        wlc = np.exp(-mass * tau)
        cumulant = first_cumulant(tau, wlc)
        np.testing.assert_allclose(cumulant[2:-2], mass, atol=1e-6)

    def test_wlc_from_positive_spectrum_stays_positive(self) -> None:
        energies = np.linspace(-1.0, 4.0, 2000)
        rho = breit_wigner_spectral_function(energies, resonance_energy=0.8, width=0.25)
        tau = np.linspace(0.01, 1.50, 20)
        wlc = wlc_from_spectrum(energies, rho, tau)
        self.assertTrue(np.all(wlc > 0.0))

    def test_tabulated_profile_interpolates(self) -> None:
        profile = TabulatedRadialProfile(
            radius=np.array([0.0, 0.5, 1.0]),
            values=np.array([1.0, 0.5, 0.0]),
            label="phi",
        )
        self.assertAlmostEqual(float(profile(0.25)), 0.75)

    def test_rate_equation_constant_gain_loss(self) -> None:
        times = np.linspace(0.0, 1.0, 11)
        occupancy = solve_rate_equation(times, 0.0, gain=2.0, loss_rate=1.0)
        self.assertGreater(occupancy[-1], occupancy[0])
        self.assertLessEqual(occupancy[-1], 2.0)

    def test_table_loader_handles_headerless_numeric_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.dat"
            path.write_text("0 1 2\n3 4 5\n", encoding="utf-8")
            table = load_table(path)
            self.assertEqual(table.header, ("col0", "col1", "col2"))
            self.assertEqual(table.data.shape, (2, 3))

    def test_table_loader_handles_one_column_numeric_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample_one_col.dat"
            path.write_text("1.0\n2.5\n3.5\n", encoding="utf-8")
            table = load_table(path)
            self.assertEqual(table.header, ("col0",))
            self.assertEqual(table.data.shape, (3, 1))
            np.testing.assert_allclose(table.data[:, 0], [1.0, 2.5, 3.5])

    def test_safe_extract_member_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            target = safe_extract_member_path(output_dir, "anc/data.txt")
            self.assertTrue(target.is_relative_to(output_dir.resolve()))
            with self.assertRaises(ValueError):
                safe_extract_member_path(output_dir, "anc/../../evil.txt")

    def test_weighted_quadratic_extrapolation_recovers_intercept(self) -> None:
        tau = np.linspace(0.01, 0.08, 8)
        values = 1.2 - 0.5 * tau + 0.3 * tau**2
        sigma = np.full_like(tau, 0.01)
        beta, covariance = weighted_quadratic_extrapolation(tau, values, sigma, n_points=8)
        self.assertAlmostEqual(float(beta[0]), 1.2, places=10)
        self.assertGreater(covariance[0, 0], 0.0)

    def test_global_common_ms_fit_reproduces_consistent_intercepts(self) -> None:
        params = {
            0.195: (0.48, 0.19, 1.10),
            0.251: (0.50, 0.19, 1.16),
            0.293: (0.53, 0.19, 1.22),
            0.352: (0.57, 0.19, 1.28),
        }
        intercepts: dict[float, dict[float, object]] = {}
        for temperature_gev, (md, ms, cb) in params.items():
            intercepts[temperature_gev] = {}
            for distance_fm in (0.224, 0.505, 0.757):
                intercepts[temperature_gev][distance_fm] = type(
                    "Estimate",
                    (),
                    {
                        "intercept": float(screened_cornell_potential(distance_fm, md, ms, cb)),
                        "intercept_sigma": 0.01,
                    },
                )()

        fits = fit_global_common_ms(intercepts)
        for temperature_gev, fit in fits.items():
            self.assertAlmostEqual(fit.ms, 0.19, places=6)
            self.assertLess(fit.chi2, 1e-8)
            self.assertAlmostEqual(fit.md, params[temperature_gev][0], places=6)
            self.assertAlmostEqual(fit.cb, params[temperature_gev][2], places=6)

    def test_static_tm_cumulant_is_anchored_to_potential(self) -> None:
        energies = np.linspace(-1.0, 3.0, 801)
        kernel = SelfEnergyKernel(
            temperature_gev=0.25,
            energies=energies,
            real_part=0.05 * np.tanh(energies),
            imag_part=-(0.08 + 0.02 * (energies + 1.0)),
        )
        tau_t = np.array([0.0, 0.1, 0.2])
        cumulant = model_cumulant_curve(
            temperature_gev=0.25,
            tau_t=tau_t,
            potential=0.6,
            phi_value=0.4,
            kernel=kernel,
        )
        self.assertAlmostEqual(float(cumulant[0]), 0.6, places=10)

    def test_static_tm_cumulant_stays_anchored_under_kernel_deformation(self) -> None:
        energies = np.linspace(-1.0, 3.0, 801)
        kernel = SelfEnergyKernel(
            temperature_gev=0.25,
            energies=energies,
            real_part=0.05 * np.tanh(energies),
            imag_part=-(0.08 + 0.02 * (energies + 1.0)),
        )
        cumulant = model_cumulant_curve(
            temperature_gev=0.25,
            tau_t=np.array([0.0, 0.1, 0.2]),
            potential=0.55,
            phi_value=0.5,
            kernel=kernel,
            re_sigma_offset=0.2,
            re_sigma_scale=1.4,
            re_sigma_slope=-0.6,
            re_sigma_curvature=0.4,
            re_sigma_radius=0.3,
            re_sigma_radius_curvature=-0.2,
            im_sigma_scale=1.7,
            im_sigma_slope=-0.3,
            im_sigma_curvature=0.5,
            im_sigma_radius=-0.4,
            im_sigma_radius_curvature=0.25,
            im_sigma_bias=0.1,
            distance_shape=0.7,
        )
        self.assertAlmostEqual(float(cumulant[0]), 0.55, places=10)

    def test_bazavov_correlator_loader_reshapes_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wilson.txt"
            rows = []
            value = 1.0
            for jack in range(3):
                for radius in range(2):
                    for tau in range(3):
                        rows.append(f"{tau + 1} {radius + 1} {value:.1f} 0")
                        value += 1.0
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            samples = load_bazavov_wilson_correlator(path, n_radius=2, n_tau=3, n_jackknife=2)
            self.assertEqual(samples.shape, (2, 3, 3))
            self.assertAlmostEqual(float(samples[0, 0, 0]), 1.0)
            self.assertAlmostEqual(float(samples[1, 2, 2]), 18.0)

    def test_bazavov_effective_mass_curve_recovers_constant_mass(self) -> None:
        lattice_spacing_fm = 0.02
        mass_gev = 3.4
        tau_index = np.arange(1, 7, dtype=float)
        correlator = np.exp(-mass_gev * lattice_spacing_fm * tau_index / 0.1973269804)
        samples = np.column_stack([correlator, correlator, correlator])
        tau_fm, meff, sigma = bazavov_effective_mass_curve(samples, lattice_spacing_fm)
        np.testing.assert_allclose(tau_fm, lattice_spacing_fm * np.arange(1, 6, dtype=float))
        np.testing.assert_allclose(meff, mass_gev, atol=1e-12)
        np.testing.assert_allclose(sigma, 0.0, atol=1e-12)

    def test_summarize_spectral_curve_recovers_peak_and_fwhm(self) -> None:
        energies = np.linspace(-1.0, 3.0, 4001)
        gamma = 0.2
        center = 0.7
        rho = gamma**2 / ((energies - center) ** 2 + gamma**2)
        summary = summarize_spectral_curve(energies, rho)
        self.assertAlmostEqual(summary["peak_energy_gev"], center, places=3)
        self.assertAlmostEqual(summary["fwhm_gev"], 2.0 * gamma, places=2)

    def test_peak_ansatz_fit_recovers_lorentzian_width(self) -> None:
        energies = np.linspace(-1.0, 3.0, 4001)
        gamma = 0.18
        center = 0.55
        rho = 0.02 + 1.7 / (1.0 + ((energies - center) / gamma) ** 2)
        fit = _fit_peak_ansatz(energies, rho, ansatz="lorentzian")
        self.assertAlmostEqual(fit["center_gev"], center, places=2)
        self.assertAlmostEqual(fit["fwhm_gev"], 2.0 * gamma, places=2)

    def test_phi_erf_ansatz_reproduces_benchmark_values(self) -> None:
        fit = PotentialFit(
            md=0.5,
            ms=0.2,
            cb=1.1,
            chi2=0.0,
            n_points=0,
            residuals=(),
            residual_sigma=(),
            phi_0224=0.11,
            phi_0505=0.53,
            phi_0757=0.78,
        )
        self.assertAlmostEqual(_phi_from_fit(fit, 0.224), 0.11, places=8)
        self.assertAlmostEqual(_phi_from_fit(fit, 0.505), 0.53, places=8)
        self.assertAlmostEqual(_phi_from_fit(fit, 0.757), 0.78, places=8)

    def test_mix_bazavov_profiles_uses_envelope_average(self) -> None:
        primary = np.array([[1.0, 10.0, 1.0], [2.0, 20.0, 2.0]])
        secondary = np.array([[1.0, 12.0, 0.5], [2.0, 18.0, 1.0]])
        value, sigma = _mix_bazavov_profiles(primary, secondary)
        np.testing.assert_allclose(value, [10.75, 19.5])
        np.testing.assert_allclose(sigma, [1.75, 2.5])

    def test_public_outer_loop_anchor_loader_returns_benchmark_temperatures(self) -> None:
        root = PROJECT_ROOT
        anchors = load_public_outer_loop_anchors(root)
        self.assertEqual(set(anchors.keys()), {0.195, 0.251, 0.293, 0.352})
        hot = anchors[0.352]
        self.assertGreater(hot.sigma_re0, 0.0)
        self.assertGreater(hot.sigma_im1, 0.0)

    def test_fit_absolute_polynomial_kernel_parameters_recovers_exact_kernel(self) -> None:
        kernel = polynomial_self_energy_kernel(
            temperature_gev=0.293,
            re_constant=0.12,
            re_slope=-0.44,
            re_curvature=0.31,
            im_log_amplitude=-1.75,
            im_slope=0.27,
            im_curvature=-0.18,
        )
        params = fit_absolute_polynomial_kernel_parameters(kernel)
        expected = (0.12, -0.44, 0.31, -1.75, 0.27, -0.18)
        np.testing.assert_allclose(params, expected, atol=1.0e-10)

    def test_tang_fig4_targets_reproduce_fig5_curves(self) -> None:
        root = PROJECT_ROOT
        fig4 = load_tang_fig4_targets(root)
        fig5 = load_tang_fig5_targets(root)
        for temperature_gev, (md, ms, cb) in fig4.items():
            radius, target = fig5[temperature_gev]
            model = screened_cornell_potential(radius, md, ms, cb)
            self.assertLess(float(np.mean(np.abs(model - target))), 5.0e-3)


if __name__ == "__main__":
    unittest.main()
