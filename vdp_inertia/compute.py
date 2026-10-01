"""Optional full recomputation; the default main.py run uses bundled results.

The accepted experiment.json records each initial point, sigma, step, seed and
pair count, including the final refined F1 step. Recomputed results go into a
separate directory, leaving the supplied publication data intact.
"""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.integrate import solve_ivp
from .config import DATA, BOXES, COORDINATES, GRID, MU, EPSILON, GRID_ODE, TIGHT, MC_THREADS
from .config import model_parameters
from .model import drift, to_lienard
from .theory import response
from .monte_carlo import simulate, coupled, summarize


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def compute_cycle():
    """Settle onto the periodic orbit, then keep one upward x=0 crossing period."""
    def crossing(t, z):
        return z[0]
    crossing.direction = 1
    orbit = solve_ivp(lambda t, z: drift(z, MU), (0, 200), [2., 0.],
                      method='DOP853', events=crossing, dense_output=True, **TIGHT)
    if not orbit.success or len(orbit.t_events[0]) < 3:
        raise RuntimeError('Could not obtain a settled limit cycle.')
    periods = np.diff(orbit.t_events[0][-3:])
    if abs(periods[1] - periods[0]) > 1e-8:
        raise RuntimeError('Limit-cycle periods have not converged.')
    return orbit.sol(np.linspace(*orbit.t_events[0][-2:], 20000)).T


def grid_row(arguments):
    """Top-level worker so grids also work with Windows process spawning."""
    name, xs, v = arguments
    results = [response((x, v), MU, BOXES[name], coordinates=COORDINATES[name], **GRID_ODE[name])
               for x in xs]
    return np.array([[r[key] for key in ('E0', 'Gamma', 'D', 'face')] for r in results])


def compute_heatmap(name, cycle, directory, grid=GRID, workers=4):
    box = BOXES[name]
    nx, nv = grid
    xe, ve = np.linspace(*box[:2], nx + 1), np.linspace(*box[2:], nv + 1)
    xs, vs = (xe[1:] + xe[:-1]) / 2, (ve[1:] + ve[:-1]) / 2
    values = {key: np.full((nv, nx), np.nan) for key in ('E0', 'Gamma', 'D', 'face')}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pool.map(grid_row, [(name, xs, v) for v in vs])
        for i, row in enumerate(rows):
            for j, key in enumerate(values):
                values[key][i] = row[:, j]
            print(f'{name}: heatmap row {i + 1}/{nv}', flush=True)
    symbol = 'y' if name == 'Ms' else 'w'
    path = cycle if name == 'Ms' else to_lienard(cycle, MU)
    metadata = dict(domain=name, box=box, epsilon=EPSILON, grid=grid,
                    coordinates=COORDINATES[name], ode=GRID_ODE[name], model=model_parameters())
    np.savez_compressed(directory / f'{name}_theory.npz', xe=xe, xs=xs,
                        **{symbol + 'e': ve, symbol + 's': vs}, **values,
                        phase_path=path, metadata=json.dumps(metadata))


def compute_case(plan, directory):
    """One selected point: deterministic response, noise sweep, step checks."""
    name = plan['domain']
    initial, box = np.array(plan['original_point']), np.array(BOXES[name])
    kind = int(name == 'Mf')
    point = to_lienard(initial, MU) if kind else initial
    exact = response(point, MU, box, coordinates=COORDINATES[name], **TIGHT)
    if not exact['smooth']:
        raise ValueError('Selected point has a corner or tangent deterministic exit.')
    case = dict(label=plan['label'], domain=name, original_point=initial.tolist(),
                point=point.tolist(), **{key: exact[key] for key in ('E0', 'Gamma', 'D', 'face')},
                sweep=[], refinement=[])
    for settings in plan['sweep']:
        times, faces = simulate(initial, box=box, kind=kind, mu=MU, **settings)
        row = dict(settings, **summarize(times, faces))
        case['sweep'].append(row)
        np.savez_compressed(directory / f'{case["label"]}_sigma_{row["sigma"]:g}.npz',
                            times=times, faces=faces, **settings)
        print(f'{case["label"]}: sigma={row["sigma"]:g}, mean={row["mean"]:.9g}', flush=True)
    for settings in plan['refinement']:
        times, faces = coupled(initial, box=box, kind=kind, mu=MU, **settings)
        coarse, fine = summarize(times[:, 0], faces[:, 0]), summarize(times[:, 1], faces[:, 1])
        delta = times[:, 1].mean(axis=1) - times[:, 0].mean(axis=1)
        difference, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))
        base = next(row for row in case['sweep'] if row['sigma'] == settings['sigma'])
        acceptable = abs(difference) <= max(3 * se, .5 * base['se'])
        case['refinement'].append(dict(settings, coarse=coarse, fine=fine,
                                      difference=difference, difference_se=se, acceptable=acceptable))
        np.savez_compressed(directory / f'{case["label"]}_check_{settings["sigma"]:g}.npz',
                            times=times, faces=faces, **settings)
    write_json(directory / f'{case["label"]}_results.json', case)
    if not all(check['acceptable'] for check in case['refinement']):
        raise RuntimeError(f'{case["label"]}: step check failed; refine h before using these results.')
    return case


def recompute(directory):
    """Repeat the experiment without overwriting the supplied results."""
    if directory.resolve() == DATA.resolve():
        raise ValueError('Choose a separate directory for recomputed results.')
    experiment = json.loads((DATA / 'experiment.json').read_text(encoding='utf-8'))
    if experiment['model'] != model_parameters():
        raise ValueError('Update experiment.json and its initial points when changing the model.')
    directory.mkdir(parents=True, exist_ok=True)
    set_num_threads(min(MC_THREADS, get_num_threads()))
    cycle = compute_cycle()
    np.savez_compressed(directory / 'cycle.npz', phase_path=cycle)
    for name in BOXES:
        compute_heatmap(name, cycle, directory)
    cases = [compute_case(plan, directory) for plan in experiment['cases']]
    write_json(directory / 'monte_carlo.json', cases)
    write_json(directory / 'experiment.json', experiment)


