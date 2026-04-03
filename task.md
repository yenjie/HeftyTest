Historical planning note.

This file records the original benchmark brief that guided the first pass of
the project. It is not synchronized with the current implementation.

For the current runnable workflows, command mapping, and branch definitions, use
[docs/code_structure.md](docs/code_structure.md).

Use one benchmark, not three.

The best first benchmark is:

Reproduce Fig. 2 of Tang, Mukherjee, Petreczky, and Rapp, "T-matrix Analysis of Static Wilson Line Correlators from Lattice QCD at Finite Temperature", and then extract the corresponding in-medium potential shown in Fig. 5 of the same paper. This is the cleanest test of "T-matrix extraction from lattice data" because the paper fits the first cumulant `m1(r,tau,T)` of Wilson-line correlators, defines it as `m1 = - d/dtau ln W`, and proves that in the `tau -> 0` limit it gives the effective potential `V_tilde(r,T)`. The target plot uses four temperatures, `T = 0.195, 0.251, 0.293, 0.352 GeV`, and three distances, `r = 0.224, 0.505, 0.757 fm`. 

Why this is the right benchmark: it isolates the actual extraction step before you add bottomonium wavefunctions, operator smearing, and pole finding. In the later bottomonium paper, the HEFTY group explicitly treats the Wilson-line correlators as an input stage inside a larger combined fit, so starting there gives you the simplest scientifically meaningful test. ([arXiv][1])

Here is the concrete task I would hand to an AI agent.

1. Read exactly these documents first.

A. Tang et al. 2023/2024, static Wilson-line T-matrix paper. The agent should read Sec. III for the observable definitions, especially
`W(r,tau,T) = integral dE exp(-E tau) rho(E,r,T)`
and
`m1(r,tau,T) = - d/dtau ln W(r,tau,T)`,
plus Appendix A for the identity `m1(r,tau=0,T) = V_tilde(r,T)`. It also needs Sec. IV.2 and Figs. 2 to 5, because those are the benchmark target and the intended fitted outputs. 

B. Bazavov et al. 2024, "Un-screened forces in Quark-Gluon Plasma?" This is the best public lattice-side companion because it studies temporal Wilson-line correlators in 2+1 flavor lattice QCD over `153 to 352 MeV` and the arXiv record includes an ancillary `data.tar.xz` archive. That archive is the first place I would tell the agent to look for machine-readable lattice inputs. ([arXiv][2])

C. Bala et al. lattice overview. This is useful as a short sanity-check document because it summarizes the lattice extraction program and notes that the dominant spectral peak position associated with the real part of the interquark potential remains essentially temperature independent on HISQ lattices. ([arXiv][3])

2. Define the required deliverables before doing any fitting.

The agent must produce exactly four files:

`plot_m1_reproduction.png`
A reproduction of Fig. 2 from Tang et al., with lattice points and model curves for all four temperatures and the three benchmark distances. 

`plot_Vtilde_extracted.png`
An extracted `V_tilde(r,T)` plot matching Fig. 5 from the same paper. 

`fit_params.json`
One entry per temperature containing `md`, `ms`, `cb`, the total chi2, and the number of fitted points. The Tang paper states that the fit parameters for this stage are exactly the screening masses `md`, `ms`, and the string-breaking control parameter `cb`. 

`report.md`
A short note listing the data source, preprocessing choices, differentiation method for `m1`, fit ranges, and a numerical check of the model identity connecting `m1(tau=0)` to `V_tilde`. 

3. Build the benchmark dataset this way.

Use the lattice Wilson-line correlator data first, not data digitized from the Tang figure. Restrict the benchmark to the four temperatures used in the Tang fit: `0.195, 0.251, 0.293, 0.352 GeV`. For the first pass, keep only the three distances shown directly in the target figure: `0.224, 0.505, 0.757 fm`. That gives a compact but nontrivial dataset. 

From the lattice correlators `W(r,tau,T)`, compute the first cumulant with
`m1(r,tau,T) = - d/dtau ln W(r,tau,T)`.
Use a local cubic spline or Savitzky-Golay differentiator, not a naive two-point derivative, because the benchmark is sensitive to both the intercept and the slope. Tang et al. explicitly compare the fit quality in terms of the `tau = 0` intercept and the slopes of `m1`. 

The Tang paper also says they use lattice `m1` with excited-state contributions subtracted, following the lattice analysis. So the agent should either use already-subtracted lattice inputs or reproduce that subtraction before fitting. Do not let it fit raw short-tau data with obvious hybrid contamination. 

4. Use this reduced model first.

Do not start with the full self-consistent light-parton plus equation-of-state loop. Tang et al. describe that full procedure, but it is too large for a first agent benchmark. For the first benchmark, isolate the static sector and fit only the Wilson-line cumulants. The same paper says the initial fit stage refines `md`, `ms`, and `cb` from `m1` of the static Wilson-line correlators. 

The model to use is the static spectral ansatz written in the paper:
`rho(E,r,T) = -(1/pi) Im [ 1 / ( E - V_tilde(r,T) - phi(r,T) Sigma_QQbar(E,T) ) ]`.
The Wilson-line correlator is then the Laplace transform of this `rho`. 

For the potential, use the explicit screened Cornell-like form given in Eq. (5) of the Tang paper:
`V_tilde(r,T) = -(4/3) alpha_s [ exp(-md r)/r + md ] - (sigma/ms) [ exp(-ms r - (cb ms r)^2 ) - 1 ]`
with starting values `alpha_s = 0.27`, `sigma = 0.225 GeV^2`, and fix the Lorentz-mixing parameter at `chi = 0.8` for this benchmark. Tang et al. state these values and that `chi = 0.8` is the choice used in the Wilson-line fit. 

For the interference function `phi(r,T)`, do not make the agent derive it from scratch on the first pass. Tell it to digitize Fig. 3 from Tang et al. once, spline it in `r` at each temperature, and keep it fixed during the fit. The paper says they interpolated `phi(r,T)` from previous work and allowed only modest variations. 

5. Give the agent these exact fit instructions.

Fit temperature by temperature first. For each of the four temperatures, optimize `md(T)`, `ms(T)`, and `cb(T)` against all available `m1(r,tau,T)` points for the three benchmark distances. Start with bounded ranges
`md in [0.2, 1.2] GeV`
`ms in [0.15, 0.30] GeV`
`cb in [1.0, 2.5]`.
Those ranges are consistent with the scales shown in Figs. 3 and 4. 

Use this loss:
`chi2 = sum over all points [ (m1_model - m1_lattice)^2 / sigma^2 ]`
plus two weak regularizers:
`lambda1 * sum_T (ms(T) - 0.2)^2`
and
`lambda2 * smoothness_in_T(md, cb)`.
This is justified because Tang et al. report that the string-screening mass can be kept roughly constant at `ms = 0.2 GeV`, while `cb` increases with temperature and `md` changes more mildly. 

After fitting each temperature separately, run one global fit over all four temperatures with the extra prior that `ms(T)` is constant or nearly constant. Accept the global fit only if it does not visibly worsen the four-panel `m1` reproduction. 

6. Grade the agent this way.

Primary pass criterion: the reproduced `m1` plot should visually track both the intercept and slope of the lattice points in all four temperature panels, especially at `r = 0.505` and `0.757 fm`, where the temperature dependence is strongest. That is exactly the fit behavior emphasized in the paper. ([ar5iv][4])

Secondary pass criterion: the extracted `V_tilde(r,T)` must satisfy the model-side identity that `m1(r,tau=0,T)` equals the potential, and the resulting curves should look like Fig. 5: weak temperature dependence at small to intermediate `r`, and less screening at higher temperature than older free-energy-based extractions. Tang et al. explicitly state that their Wilson-line fit leads to less screening at higher temperatures and larger distances. 

Parameter sanity criterion: a passing fit should come out with `ms` roughly flat near `0.2 GeV`, `cb` increasing with `T`, and `md` varying more gently. If the agent returns rapidly increasing `ms(T)` or strongly oscillatory `cb(T)`, it probably overfit numerical derivative noise rather than learning the extraction. 

7. After the agent passes this, move to one extension only.

The next benchmark should be the bottomonium extension in the 2024 HEFTY paper: reproduce its Fig. 2 effective masses and Fig. 3 in-medium potentials. That paper explicitly says the fit is a combined one over the equation of state, static Wilson-line correlators, and bottomonium correlators with extended operators, so it is the natural level-2 test after the static benchmark. ([arXiv][1])

For a grading-friendly level-3 extension, the 2025 spectroscopy paper is useful because its arXiv page exposes ancillary `Figure1.dat`, `Figure2_195MeV.dat`, `Figure2_293MeV.dat`, `Figure2_352MeV.dat`, and `Figure3_251MeV.dat` files. Those are ideal for automated comparison once the agent is ready for pole finding in the complex plane. ([arXiv][5])

So the short version is: make the agent reproduce Tang Fig. 2 first, require it to output Fig. 5 second, and do not let it touch bottomonium until it can stably recover `m1` and `V_tilde` from Wilson-line lattice data.

[1]: https://arxiv.org/html/2411.09132v1 "Bottomonium Properties in QGP from a Lattice-QCD Informed -Matrix Approach"
[2]: https://arxiv.org/abs/2308.16587 "[2308.16587] Un-screened forces in Quark-Gluon Plasma?"
[3]: https://arxiv.org/abs/2211.12937 "[2211.12937] Static quark anti-quark interactions at non-zero temperature from lattice QCD"
[4]: https://ar5iv.labs.arxiv.org/html/2310.18864 "[2310.18864] -matrix Analysis of Static Wilson Line Correlators from Lattice QCD at Finite Temperature"
[5]: https://arxiv.org/abs/2502.09044 "[2502.09044] Quarkonium Spectroscopy in the Quark-Gluon Plasma"
