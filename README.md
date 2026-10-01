# Van der Pol stochastic inertia

Code and numerical data for the van der Pol example in the manuscript.
The repository is self-contained: it does not import the earlier exploratory
studies or require the manuscript sources.

## Reproduce the figures

Use Python 3.12 with the versions recorded in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

On macOS/Linux, use `.venv/bin/python`. With dependencies already installed:

```text
python main.py
```

This reads the bundled results and writes the five manuscript PDFs to `figures/`:

- `phase_portrait_original_coordinates.pdf`
- `phase_portrait_lienard_coordinates.pdf`
- `heatmaps_comparison.pdf`
- `Ms_exit_time_sigma.pdf`
- `Mf_exit_time_sigma.pdf`

No simulations, display or LaTeX installation are needed for this command.
Default paths are relative to the repository, so the script also works when
called from another directory. Optional exports and a different destination:

```text
python main.py --formats pdf png svg --output-dir results/figures
```

## Files

```text
main.py                 Figure-generation entry point
verify.py               Numerical regression and pipeline checks
requirements.txt        Tested dependency versions
vdp_inertia/
    config.py           Model, domains, paths and solver settings
    model.py            Drift, derivatives and coordinate transformation
    theory.py           First exit and backward derivative transport
    monte_carlo.py      Stochastic Heun and coupled step refinement
    compute.py          Full experiment orchestration
    plot.py             The five manuscript figures and data checks
data/
    cycle.npz           Settled periodic orbit in original coordinates
    Ms_theory.npz       Slow-domain deterministic grid
    Mf_theory.npz       Fast-domain deterministic grid
    monte_carlo.json    Accepted means, standard errors and refinement checks
    experiment.json     Initial points, noise levels, time steps, seeds, pair counts
    validation.json     Historical numerical checks and grid provenance
figures/                Five publication PDFs
```

Edit `vdp_inertia/plot.py` for figure appearance. Historical scripts, local
environments, duplicate exports and temporary runs are not part of the
publication repository. New computation outputs go in the ignored `results/`
directory. The source data in `data/` are preserved.

## Model and coordinates

In original time, epsilon = 0.1 and mu = 1/epsilon = 10:

```text
dx = y dt + sqrt(2 sigma) dW1
dy = [mu (1 - x^2) y - x] dt + sqrt(2 sigma) dW2
w  = epsilon y + x^3/3 - x
```

The Brownian motions are independent. The slow domain is
`(-2.1, -0.8) x (-2, 2)` in `(x,y)`. The fast domain is the inverse image of
`(-0.5, 1.1) x (-0.95, -0.5)` under the map to `(x,w)`. Every boundary component
absorbs. The portraits and grids describe these same physical domains.

All stochastic trajectories are simulated in original coordinates. The fast
**deterministic** calculation uses Lienard coordinates, applying the original
noise generator through its complete chain rule:

```text
Gamma = E0_xx + 2(x^2-1) E0_xw
        + [(x^2-1)^2 + epsilon^2] E0_ww + 2x E0_w.
```

The final term is essential: the transformed Ito drift has correction
`sigma (0, 2x)`. In the slow domain, Gamma is simply the original Laplacian.
The response is `D = integral_0^T Gamma(flow(t,z)) dt`. It is computed from
backward transport of the exit-time gradient and Hessian along a stored forward
trajectory. Terminal derivatives use the planar exit face in the calculation's
coordinates. Corner or tangent exits receive NaN derivatives.

The plotted tangents are `E0 + sigma D`; their slopes are not fitted to Monte
Carlo data. F2 has approximately zero first-order response, which does not
establish absence of a higher-order noise effect. Heatmap panels use separate
symmetric logarithmic scales, without clipping extrema. The critical curve is
`w=x^3/3-x`; its pullback at fixed epsilon is `y=0`. The coordinate map itself
degenerates at epsilon = 0.

## Simulation and data provenance

Each mean uses 25,000 independent antithetic pairs (50,000 paths). Pair means
are used to calculate standard errors. The archive contains 72 positive
point/noise combinations, or 3.6 million production paths. The figures show 12
positive levels per slow point and 10 per fast point; sigma = 0 is a
deterministic endpoint. The additional sigma = 0.2 results remain in the data.
The JSON records standard errors even though the figures omit error bars.
Individual production paths are not bundled; they can be regenerated.

The solver uses additive-noise stochastic Heun in original coordinates, with
an inward boundary shift of `0.5825971579390107 sqrt(2 sigma h)` on x/y faces.
For w faces, multiply this by `sqrt((x^2-1)^2+epsilon^2)`. The nonlinear w
coordinate is evaluated at every step. Censored paths cause an error instead
of being silently excluded.

Coupled h versus h/2 checks share Brownian increments. Their acceptance rule is
`abs(mean difference) <= max(3 SE_difference, 0.5 SE_production)`. All 18 archived
final checks passed. These selected-level checks are numerical diagnostics,
not rigorous error bounds for every plotted point.

The exact accepted steps and seeds, including the refined F1 run, are in
`experiment.json`. Historical labels F-, F0 and F+ mean F1, F2 and F3.
`validation.json` preserves earlier validation records, including their original
labels and metadata. The bundled slow grid used maximum ODE step 0.1 and the
fast grid 0.05; `config.py` preserves these per-domain settings. Selected-point
predictions use tighter tolerances. Recomputed floating-point results can vary
slightly across solver/library versions.

## Verify or recompute

```text
python verify.py
python verify.py --report results/verification.json
```

Verification checks all six predictions, original-coordinate finite-difference
Laplacians, a genuine curved-domain exit, zero-noise simulations, seeded results
across thread counts, invalid inputs and a small parallel recomputation. Smoke
outputs are temporary and removed automatically; it does not repeat the full
Monte Carlo sweep.

To repeat the full numerical experiment and plot it:

```text
python main.py --recompute
```

This is expensive. It writes data (including raw paired exit times and faces)
to `results/data/` and figures to `results/figures/`. Failed refinement checks
stop the run and leave diagnostics for further refinement. The bundled data
are never overwritten. To redraw a completed recomputation:

```text
python main.py --data-dir results/data --output-dir results/figures
```

When changing the physical model, update both `config.py` and the experiment
specification, including initial points. Data/model mismatches are rejected.
The checked-in PDFs reproduce the current manuscript experiment.
