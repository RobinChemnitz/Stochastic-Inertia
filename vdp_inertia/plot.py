"""Publication figures from saved data; no numerical simulations run here."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from .config import BOXES, COORDINATES, EPSILON, MU, model_parameters
from .model import cubic, drift, to_lienard

FORMATS = ('pdf',)
DPI = 400
PHASE_SIZE = (7.1, 5.4)
HEATMAP_SIZE = (6, 3.8)  # per panel, including its colorbar
SIGMA_SIZE = (3.5, 2.8)
TANGENT_END = 0.02
SIGMA_MAX = 0.15
DOMAIN_COLORS = {'Ms': '#D55E00', 'Mf': '#8C56A2'}
NAMES = {'Ms': 'Slow region', 'Mf': 'Fast region'}
POINT_STYLES = {'S1': ('#0072B2', 'o'), 'S2': ('#D55E00', 's'),
                'S3': ('#009E73', '^'), 'F1': ('#009E73', '^'),
                'F2': ('#D55E00', 's'), 'F3': ('#0072B2', 'o')}
ARROW_GRID = (9, 8)
ARROW_LENGTH = 0.048


def load_data(directory):
    """Check the experiment identity before overlaying data and geometry."""
    metadata = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
    if metadata['model'] != model_parameters():
        raise ValueError('Saved data do not match config.py. Recompute the experiment.')
    with np.load(directory / 'cycle.npz') as saved:
        cycle = saved['phase_path'].copy()
    fields = {}
    for name, box in BOXES.items():
        with np.load(directory / f'{name}_theory.npz') as saved:
            fields[name] = {key: saved[key] for key in saved.files}
        meta = json.loads(str(fields[name]['metadata']))
        if tuple(meta['box']) != box or meta['epsilon'] != EPSILON:
            raise ValueError(f'{name}: heatmap data do not match config.py.')
    cases = json.loads((directory / 'monte_carlo.json').read_text(encoding='utf-8'))
    if sorted(c['label'] for c in cases) != sorted(POINT_STYLES):
        raise ValueError('Expected three selected points in each regime.')
    plans = {plan['label']: plan for plan in metadata['cases']}
    for case in cases:
        plan = plans[case['label']]
        if case['domain'] != plan['domain'] or not np.allclose(
                case['original_point'], plan['original_point'], atol=1e-12, rtol=0):
            raise ValueError('Selected points do not match experiment.json.')
        point = np.asarray(case['original_point'])
        if case['domain'] == 'Mf':
            point = to_lienard(point, MU)
        if not np.allclose(point, case['point'], atol=1e-12, rtol=0):
            raise ValueError('Inconsistent selected-point coordinates.')
        if any(row['censored'] for row in case['sweep']):
            raise ValueError('Monte Carlo data contain censored paths.')
        if any(not check['acceptable'] for check in case['refinement']):
            raise ValueError('Monte Carlo data contain a failed step-refinement check.')
    return cycle, fields, cases


def set_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 14, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42})


def save(fig, stem, directory, formats=FORMATS):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(directory / f'{stem}.{extension}', dpi=DPI,
                    bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


def math_label(label):
    return rf'${label[0]}_{{{label[1:]}}}$'


def mark_folds(ax, coordinates, box=None):
    """Black dots at folds. Landing points are deliberately unmarked."""
    for x in (-1., 1.):
        v = 0. if coordinates == 'original' else cubic(x)
        if box is None or (box[0] < x < box[1] and box[2] < v < box[3]):
            ax.scatter(x, v, color='black', marker='o', s=32, zorder=8)


def phase_portrait(cycle, coordinates, directory, formats=FORMATS):
    fig, ax = plt.subplots(figsize=PHASE_SIZE, layout='constrained')
    for name, box in BOXES.items():
        x = np.linspace(box[0], box[1], 700)
        lower, upper = np.full_like(x, box[2]), np.full_like(x, box[3])
        if name == 'Ms' and coordinates == 'lienard':
            lower, upper = EPSILON * lower + cubic(x), EPSILON * upper + cubic(x)
        if name == 'Mf' and coordinates == 'original':
            lower, upper = (lower - cubic(x)) / EPSILON, (upper - cubic(x)) / EPSILON
        color = DOMAIN_COLORS[name]
        ax.fill_between(x, lower, upper, color=color, alpha=0.22, zorder=0)
        ax.plot(x, lower, color=color, alpha=0.6, lw=1.2)
        ax.plot(x, upper, color=color, alpha=0.6, lw=1.2)
        for j in (0, -1):
            ax.plot([x[j], x[j]], [lower[j], upper[j]], color=color, alpha=0.6, lw=1.2)
        mid = len(x) // 2
        offset = 0.11 if coordinates == 'lienard' else 1.1
        ax.text(x[mid], upper[mid] + offset, math_label(name),
                ha='center', color=color, fontsize=16)
    path = cycle if coordinates == 'original' else to_lienard(cycle, MU)
    # The reference critical manifold w=f(x) pulls back to y=0 at fixed epsilon.
    for left, right, style, color in [(-2.25, -1, '-', '#0072B2'),
                                     (-1, 1, '--', '#B23A48'),
                                     (1, 2.25, '-', '#0072B2')]:
        x = np.linspace(left, right, 400)
        v = cubic(x) if coordinates == 'lienard' else np.zeros_like(x)
        ax.plot(x, v, style, color=color, lw=1.4, zorder=3)
    ax.plot(path[:, 0], path[:, 1], color='.2', lw=1.8, zorder=4)
    cycle_arrows(ax, path, coordinates)
    mark_folds(ax, coordinates)
    ax.set(xlim=(-2.35, 2.3), ylim=(-15, 15) if coordinates == 'original' else (-1.6, 1.3),
           xlabel='$x$', ylabel='$y$' if coordinates == 'original' else '$w$',
           xticks=[-2, -1, 0, 1, 2])
    ax.grid(alpha=0.13)
    save(fig, f'phase_portrait_{coordinates}_coordinates', directory, formats)


def cycle_arrows(ax, path, coordinates):
    """One arrow on each slow branch, two double-arrow groups on each jump."""
    scale = np.array([4.6, 30. if coordinates == 'original' else 2.8])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0) / scale, axis=1))]

    def at(s):
        return [np.interp(s % arc[-1], arc, path[:, k]) for k in (0, 1)]

    targets = [((-1.65, 0.1), False), ((1.65, -0.1), False),
               ((0, 0.75), True), ((1.6, 0.75), True),
               ((0, -0.75), True), ((-1.6, -0.75), True)]
    for target, fast in targets:
        if coordinates == 'lienard':
            j = np.argmin(np.sum(((path - target) / scale)**2, axis=1))
        else:
            mask = abs(path[:, 1]) > 2 if fast else abs(path[:, 1]) < 1
            ids = np.flatnonzero((np.sign(target[1]) * path[:, 1] > 0) & mask)
            j = ids[np.argmin(abs(path[ids, 0] - target[0]))]
        for offset in (-0.014, 0.014) if fast else (0.,):
            s = arc[j] + offset
            ax.annotate('', xy=at(s), xytext=at(s - 0.02), zorder=5,
                        arrowprops=dict(arrowstyle='-|>', color='.2', lw=1.5,
                                        mutation_scale=14, shrinkA=0, shrinkB=0))


def heatmap_panel(fig, cell, name, key, data, cases):
    box = BOXES[name]
    coordinates = COORDINATES[name]
    symbol = 'y' if coordinates == 'original' else 'w'
    inner = cell.subgridspec(1, 2, width_ratios=[1, 0.045], wspace=0.06)
    ax, cax = fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])
    values = data[key]
    threshold = 10.**np.floor(np.log10(max(np.nanmedian(abs(values)) / 10, 1e-8)))
    bound = max(threshold, float(np.nanmax(abs(values))))
    mesh = ax.pcolormesh(data['xe'], data[symbol + 'e'], values, cmap='RdBu_r',
                        norm=SymLogNorm(threshold, vmin=-bound, vmax=bound), rasterized=True)
    xr, vr = box[1] - box[0], box[3] - box[2]
    xx, vv = np.meshgrid(np.linspace(box[0] + .06*xr, box[1] - .06*xr, ARROW_GRID[0]),
                         np.linspace(box[2] + .07*vr, box[3] - .07*vr, ARROW_GRID[1]))
    velocity = drift(np.stack([xx, vv], axis=-1), MU, coordinates)
    length = np.linalg.norm(velocity / [xr, vr], axis=-1)
    ax.quiver(xx, vv, ARROW_LENGTH * velocity[..., 0] / length,
              ARROW_LENGTH * velocity[..., 1] / length, angles='xy', scale_units='xy',
              scale=1, color='black', alpha=.8, width=.004, headwidth=3.5,
              headlength=4.5, pivot='mid', zorder=2)
    path = data['phase_path']
    ax.plot(path[:, 0], path[:, 1], 'k', lw=1.8, zorder=4)
    x = np.linspace(box[0], box[1], 600)
    ax.plot(x, np.zeros_like(x) if coordinates == 'original' else cubic(x),
            'k--', lw=1.2, zorder=3)
    mark_folds(ax, coordinates, box)
    for case in cases:
        if case['domain'] != name:
            continue
        color, marker = POINT_STYLES[case['label']]
        ax.scatter(*case['point'], s=48, marker=marker, facecolor=color,
                   edgecolor='black', linewidth=.8, zorder=10)
        ax.annotate(math_label(case['label']), case['point'], xytext=(6, 11),
                    textcoords='offset points', fontsize=10, zorder=11,
                    bbox=dict(boxstyle='round,pad=.13', fc='white', ec='none', alpha=.9))
    ax.set(xlim=box[:2], ylim=box[2:], xlabel='$x$', ylabel=f'${symbol}$',
           title=math_label(name) + ': ' + NAMES[name])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        axis.set_major_formatter(StrMethodFormatter('{x:g}'))
    bar = fig.colorbar(mesh, cax=cax)
    low, high = int(np.floor(np.log10(threshold))), int(np.floor(np.log10(bound)))
    ticks = 10.**np.array(sorted({low, (low + high) // 2, high}))
    bar.set_ticks(np.r_[-ticks[::-1], 0., ticks])
    quantity = r'\Gamma' if key == 'Gamma' else r'\mathcal{D}'
    bar.set_label(rf'${quantity}(x,{symbol})$')


def heatmaps(fields, cases, directory, formats=FORMATS):
    """The manuscript's four-panel comparison of Gamma and D in both regimes."""
    fig = plt.figure(figsize=(2 * HEATMAP_SIZE[0], 2 * HEATMAP_SIZE[1]),
                     layout='constrained')
    grid = fig.add_gridspec(2, 2)
    for row, name in enumerate(('Ms', 'Mf')):
        for col, key in enumerate(('Gamma', 'D')):
            heatmap_panel(fig, grid[row, col], name, key, fields[name], cases)
    save(fig, 'heatmaps_comparison', directory, formats)


def exit_times(cases, directory, formats=FORMATS):
    """Plot the saved Monte Carlo means and independently computed tangents."""
    for name in BOXES:
        fig, ax = plt.subplots(figsize=SIGMA_SIZE, layout='constrained')
        selected = sorted((c for c in cases if c['domain'] == name), key=lambda c: c['label'])
        for case in selected:
            color, marker = POINT_STYLES[case['label']]
            rows = [r for r in case['sweep'] if r['sigma'] <= SIGMA_MAX]
            sigmas = [0.] + [r['sigma'] for r in rows]
            means = [case['E0']] + [r['mean'] for r in rows]
            ax.plot(sigmas, means, '-', marker=marker, color=color, markersize=3.6,
                    markeredgecolor='white', markeredgewidth=.25, lw=1.25,
                    label=math_label(case['label']), zorder=3)
            s = np.linspace(0., TANGENT_END, 100)
            ax.plot(s, case['E0'] + s * case['D'], linestyle=(0, (4, 2.5)),
                    color='black', lw=1.25, zorder=6)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='black', linestyle=(0, (4, 2.5)), lw=1.25))
        labels.append(r'$E_0+\sigma D$')
        ax.legend(handles, labels, loc='upper right', ncol=2, columnspacing=.7,
                  fontsize=8, framealpha=.95, edgecolor='.85', borderpad=.35,
                  labelspacing=.3, handlelength=1.7, handletextpad=.45)
        ax.set(xlim=(-.002, SIGMA_MAX + .002), xlabel=r'$\sigma$',
               ylabel=rf'$E_\sigma^{{M_{name[-1]}}}$',
               title=math_label(name) + ': ' + ('slow regime' if name == 'Ms' else 'fast regime'))
        ax.set_ylim(bottom=0)
        if name == 'Mf':
            ax.set_ylim(0, .4)
        ax.set_xticks([0, .05, .1, .15])
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.set_yticks([0, 2, 4, 6, 8, 10] if name == 'Ms' else [0, .1, .2, .3, .4])
        ax.grid(alpha=.15, lw=.45)
        ax.tick_params(labelsize=8, length=3, width=.7)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.title.set_fontsize(10)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
        save(fig, name + '_exit_time_sigma', directory, formats)
