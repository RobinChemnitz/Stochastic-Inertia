"""Numerical regression and pipeline checks: python verify.py [--report FILE].

Smoke-test files live in a temporary directory and are removed automatically.
The bundled publication data are never modified.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numba import get_num_threads, set_num_threads

from vdp_inertia.config import DATA, MU, BOXES, COORDINATES, TIGHT, MC_THREADS
from vdp_inertia.model import to_lienard
from vdp_inertia.theory import response, exit_trajectory
from vdp_inertia.monte_carlo import simulate, coupled, summarize
from vdp_inertia.compute import compute_cycle, compute_heatmap, compute_case, write_json
from vdp_inertia.plot import load_data


def verify():
    threads = min(MC_THREADS, get_num_threads())
    set_num_threads(threads)
    _, _, cases = load_data(DATA)
    results = []
    for case in cases:
        name = case['domain']
        result = response(case['point'], MU, BOXES[name], coordinates=COORDINATES[name], **TIGHT)
        error = max(abs(result[k] - case[k]) / max(1., abs(case[k])) for k in ('E0', 'Gamma', 'D'))
        assert error < 2e-8, (case['label'], result)
        h = 1e-5 if name == 'Ms' else 1e-6
        times, faces = simulate(np.array(case['original_point']), 0., h, 2,
                                np.array(BOXES[name]), int(name == 'Mf'), 1, mu=MU)
        assert np.max(abs(times - case['E0'])) < 1.05*h
        assert np.all(faces == case['face'])
        results.append(dict(label=case['label'], scaled_error=error))

    # Perturb original x,y even for the Lienard domain: this checks the Ito term.
    for label in ('S2', 'F3'):
        case = next(c for c in cases if c['label'] == label)
        name, h = case['domain'], .0003

        def time_at(original):
            point = to_lienard(original, MU) if name == 'Mf' else original
            return exit_trajectory(point, MU, BOXES[name], coordinates=COORDINATES[name], **TIGHT)[1]

        z = np.array(case['original_point'])
        laplacian = (sum(time_at(z+h*e) + time_at(z-h*e) for e in np.eye(2)) - 4*time_at(z)) / h**2
        assert abs(laplacian - case['Gamma']) / max(1., abs(case['Gamma'])) < 2e-4

    # Check an actual curved-domain exit, beyond the six selected x-face exits.
    history = json.loads((DATA / 'validation.json').read_text(encoding='utf-8'))
    curved = history['records']['Mf_summary']['checks'][0]
    result = response(curved['point'], MU, BOXES['Mf'], coordinates='lienard', **TIGHT)
    assert result['face'] == 2
    for key in ('E0', 'Gamma', 'D'):
        np.testing.assert_allclose(result[key], curved['tight'][key], rtol=2e-7, atol=1e-9)

    # Thread count must not change the Brownian path assigned to a pair.
    point = np.array(next(c for c in cases if c['label'] == 'F3')['original_point'])
    box = np.array(BOXES['Mf'])
    settings = dict(sigma=.01, h=1e-5, n_pairs=8, box=box, kind=1, seed=1234, mu=MU)
    for solver in (simulate, coupled):
        set_num_threads(1)
        serial = solver(point, **settings)
        set_num_threads(min(2, threads))
        parallel = solver(point, **settings)
        for a, b in zip(serial, parallel):
            np.testing.assert_array_equal(a, b)
        try:
            solver(point, **dict(settings, h=0.))
        except ValueError:
            pass
        else:
            raise AssertionError('A zero time step must be rejected.')
        try:
            solver(np.array([2., 0.]), **settings)
        except ValueError:
            pass
        else:
            raise AssertionError('A start outside the corrected domain must be rejected.')
    set_num_threads(threads)
    try:
        summarize(np.ones((2, 2)), np.full((2, 2), -1))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Censored paths must not enter reported means.')

    # Exercise orbit generation, parallel grid workers, saved files and MC orchestration.
    with TemporaryDirectory(prefix='vdp-verification-') as temporary:
        directory = Path(temporary)
        cycle = compute_cycle()
        for name in BOXES:
            compute_heatmap(name, cycle, directory, grid=(3, 3), workers=2)
        experiment = json.loads((DATA / 'experiment.json').read_text(encoding='utf-8'))
        smoke_cases = []
        for plan in experiment['cases']:
            settings = next(r for r in plan['sweep'] if r['sigma'] == .01)
            small = dict(plan, sweep=[dict(settings, n_pairs=16)], refinement=[])
            smoke_cases.append(compute_case(small, directory))
        np.savez_compressed(directory / 'cycle.npz', phase_path=cycle)
        write_json(directory / 'experiment.json', experiment)
        write_json(directory / 'monte_carlo.json', smoke_cases)
        load_data(directory)
    print('Passed: six predictions, finite differences, curved exit, zero-noise exits, '
          'seed reproducibility, input guards and recomputation smoke checks.')
    return dict(predictions=results, curved_exit=True, seeded_threads=True,
                input_guards=True, pipeline_smoke=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='optional JSON verification report')
    args = parser.parse_args()
    result = verify()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, result)
