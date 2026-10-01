"""The experiment. Plot appearance is controlled separately in plot.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'
EPSILON = 0.1
MU = 1 / EPSILON
BOXES = {'Ms': (-2.1, -0.8, -2., 2.), 'Mf': (-0.5, 1.1, -0.95, -0.5)}
COORDINATES = {'Ms': 'original', 'Mf': 'lienard'}
GRID = (65, 65)
# Match the settings recorded in the bundled grids.
GRID_ODE = {
    'Ms': dict(rtol=1e-9, atol=1e-11, max_step=0.1),
    'Mf': dict(rtol=1e-9, atol=1e-11, max_step=0.05),
}
TIGHT = dict(rtol=2e-12, atol=2e-14, max_step=0.02)
MC_THREADS = 8


def model_parameters():
    return dict(epsilon=EPSILON, boxes={k: list(v) for k, v in BOXES.items()},
                coordinates=COORDINATES,
                noise='sqrt(2*sigma) independent additive noise in original x,y')
