# Task 1 Reading Notes

This note completes task 1 from [task.md](../task.md): read the three benchmark documents first and extract the parts needed for the first Wilson-line benchmark.

## Documents read

Local copies:

- [papers/2310.18864.pdf](/raid5/data/yjlee/hefty/papers/2310.18864.pdf)
- [papers/2308.16587.pdf](/raid5/data/yjlee/hefty/papers/2308.16587.pdf)
- [papers/2211.12937.pdf](/raid5/data/yjlee/hefty/papers/2211.12937.pdf)

Primary sources:

- Tang et al., `$T$-matrix Analysis of Static Wilson Line Correlators from Lattice QCD at Finite Temperature`
  `https://arxiv.org/abs/2310.18864`
  HTML reference used: `https://ar5iv.labs.arxiv.org/html/2310.18864`
- Bazavov et al., `Un-screened forces in Quark-Gluon Plasma?`
  `https://arxiv.org/abs/2308.16587`
  HTML reference used: `https://arxiv.org/html/2308.16587v2`
- Parkar, Bala et al., `Static quark anti-quark interactions at non-zero temperature from lattice QCD`
  `https://arxiv.org/abs/2211.12937`

## A. Tang et al. 2023/2024: exact benchmark target

### Observable definitions from Sec. III

The static Wilson-line correlator is the Laplace transform of the static spectral function,

`W(r,tau,T) = integral dE exp(-E tau) rho_QQbar(E,r,T)`.

In the T-matrix model, the static spectral function is written as

`rho(E,r,T) = -(1/pi) Im [ 1 / ( E - V_tilde(r,T) - phi(r,T) Sigma_QQbar(E,T) ) ]`.

Tang et al. then define the cumulants of the Wilson-line correlator. The first cumulant is the effective mass used in the lattice benchmark,

`m1(r,tau,T) = - d/dtau ln W(r,tau,T)`.

The paper explicitly states that the first benchmark quantity is `m1`, and that the second cumulant controls the slope of `m1` and therefore the width information in the spectral function.

### Key consequence from Appendix A

Appendix A proves the identity that the zero-time limit of the first cumulant recovers the effective static potential:

`m1(r,tau=0,T) = V_tilde(r,T)`.

The derivation is by expanding `W(r,tau)` around `tau = 0`, using the normalization of the spectral function and the analyticity of the two-body selfenergy. This identity is the formal reason why reproducing Fig. 2 is already a legitimate extraction benchmark for Fig. 5.

### What Sec. IV.2 says about the fit

The paper states that the first self-consistency loop refines the in-medium potential by fitting the first cumulants of the static Wilson-line correlators. The fit parameters at this stage are exactly:

- `md` for color-Coulomb screening
- `ms` for confining screening
- `cb` for string breaking / saturation

This matches the required contents of `fit_params.json` in [task.md](../task.md).

### What Figs. 2 to 5 mean for the first benchmark

- Fig. 2 is the direct target:
  first cumulants `m1(r,tau,T)` versus imaginary time for several temperatures and distances.
- Fig. 3 provides the interference function `phi(r,T)`.
  Tang et al. do not refit it from scratch here; they interpolate it from previous work based on lattice free-energy constraints.
- Fig. 4 shows the fitted parameter trends.
  The important qualitative result is:
  `ms` can be kept approximately constant at `0.2 GeV`,
  `cb` needs to increase with temperature,
  and `md` changes more mildly.
- Fig. 5 is the extracted in-medium potential.
  Compared to older extractions based on free energies, the Wilson-line constrained potential shows less screening at higher temperature and larger distance.

### Model constants needed immediately

From the Tang paper:

- `alpha_s = 0.27`
- `sigma = 0.225 GeV^2`
- use the screened Cornell-like form from Eq. (5)
- use Lorentz mixing parameter `chi = 0.8` in the Wilson-line fit

This is the exact reduced-model starting point described in [task.md](../task.md).

### Practical consequence for the benchmark code

The Tang paper explicitly says it uses lattice `m1` with excited-state contributions subtracted, following the lattice analysis. Therefore the first-pass fit should use subtracted effective-mass data, not raw short-`tau` Wilson-line data.

## B. Bazavov et al. 2024: best public lattice-side companion

### Why this is the right lattice input paper

Bazavov et al. study temporal Wilson-line correlators in `2+1` flavor HISQ lattice QCD over `153 MeV <= T <= 352 MeV`, using several lattice spacings and large temporal extent. This is the correct lattice-side companion for the Tang benchmark because it isolates the same Wilson-line extraction problem and exposes a public ancillary archive.

The arXiv abstract page includes:

- ancillary file `data.tar.xz`

which is already downloaded locally and extracted under [lattice/bazavov_2308.16587v2](/raid5/data/yjlee/hefty/lattice/bazavov_2308.16587v2).

### Lattice-analysis details that matter for the fit

Bazavov et al. define the effective mass as

`meff(tau,r,T) = - d/dtau ln W(tau,r,T)`.

They then separate the zero-temperature high-energy part and define a subtracted correlator,

`W_sub(tau,r,T) = W(tau,r,T) - W_high(tau,r)`.

This subtraction removes the `T = 0` high-energy contribution and isolates the medium-dependent spectral pieces. The paper states that the non-monotonic short-`tau` behavior induced by gradient-flow smearing is absent in the subtracted effective masses. This is exactly why the subtracted data are the correct starting point for the Tang benchmark.

The paper also states that at small `tau`, the subtracted effective mass decreases approximately linearly, consistent with a broadened dominant peak. That is directly compatible with the Tang benchmark, which emphasizes both the `tau = 0` intercept and the slope of `m1`.

### Main physics conclusion relevant here

Bazavov et al. find that the real part of the potential is effectively temperature independent over the range they study and is therefore unscreened in the sense relevant to this benchmark. This is the lattice-side result that Tang’s T-matrix extraction is trying to reproduce microscopically.

### Concrete local mapping for the first benchmark

Using the Bazavov ancillary package, the clean first subset is already assembled in [lattice/benchmark_wlc](/raid5/data/yjlee/hefty/lattice/benchmark_wlc).

For the benchmark temperatures, the relevant ensemble is:

- `beta = 8.249`
- `Nt = 36, 28, 24, 20`

with the temperature mapping:

- `Nt = 36 -> T = 0.195 GeV`
- `Nt = 28 -> T = 0.251 GeV`
- `Nt = 24 -> T = 0.293 GeV`
- `Nt = 20 -> T = 0.352 GeV`

Using `a = 0.1973269804 / (Nt * T)` gives `a ~= 0.0281 fm`, so the Tang distances map to:

- `r/a = 8 -> r ~= 0.224 fm`
- `r/a = 18 -> r ~= 0.505 fm`
- `r/a = 27 -> r ~= 0.757 fm`

The primary benchmark inputs are therefore the 12 subtracted files in:

- [lattice/benchmark_wlc/data_subtracted](/raid5/data/yjlee/hefty/lattice/benchmark_wlc/data_subtracted)

with matching raw files and Bazavov fit curves stored for cross-checks.

## C. Bala / Parkar et al. overview: short sanity-check reading

This short review emphasizes that Wilson-line correlators on realistic HISQ lattices support a temperature-independent position of the dominant spectral peak associated with the real part of the interquark potential.

Its abstract says, in substance:

- spectral information is extracted from Wilson-line correlators in Coulomb gauge
- four complementary methods were used
- on HISQ lattices, the position of the dominant spectral peak associated with the real part of the interquark potential remains unaffected by temperature

This is the clean sanity check for the first benchmark:

- if the fit strongly screens `V_tilde(r,T)` already at moderate distance,
  it is likely inconsistent with the modern lattice picture
- if the extracted `m1(tau=0)` and `V_tilde` stay close to the zero-temperature potential at small and intermediate `r`,
  that is qualitatively in line with the current lattice understanding

## What task 1 implies for implementation

The fitting stage should now proceed with these assumptions:

1. Use the subtracted Bazavov effective-mass data as the default lattice input.
2. Restrict the first pass to `T = 0.195, 0.251, 0.293, 0.352 GeV`.
3. Restrict the first pass to `r = 0.224, 0.505, 0.757 fm`, i.e. lattice indices `r/a = 8, 18, 27`.
4. Fit only `md`, `ms`, and `cb` in the reduced static-sector model.
5. Keep `phi(r,T)` fixed from Tang Fig. 3 on the first pass.
6. Check both the `tau = 0` intercept and the slope of `m1`.
7. Demand consistency with `m1(r,tau=0,T) = V_tilde(r,T)`.

## Files prepared as a result of task 1

- paper PDFs:
  [papers/2310.18864.pdf](/raid5/data/yjlee/hefty/papers/2310.18864.pdf),
  [papers/2308.16587.pdf](/raid5/data/yjlee/hefty/papers/2308.16587.pdf),
  [papers/2211.12937.pdf](/raid5/data/yjlee/hefty/papers/2211.12937.pdf)
- lattice benchmark subset:
  [lattice/benchmark_wlc](/raid5/data/yjlee/hefty/lattice/benchmark_wlc)

Task 1 is therefore complete in the sense intended by [task.md](../task.md): the exact documents have been read and converted into implementation-ready notes for the first benchmark.
