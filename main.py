"""Generate the five manuscript figures from bundled data: python main.py."""

import argparse
from pathlib import Path

from vdp_inertia.config import DATA, FIGURES, RESULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--recompute', action='store_true',
                      help='repeat the full experiment in results/ (expensive)')
    mode.add_argument('--data-dir', type=Path,
                      help='plot an existing recomputation instead of bundled data')
    parser.add_argument('--output-dir', type=Path,
                        help='figure destination (default: figures/, or results/figures/)')
    parser.add_argument('--formats', nargs='+', choices=('pdf', 'png', 'svg'),
                        default=['pdf'], help='output formats (default: pdf)')
    args = parser.parse_args()

    data_directory = args.data_dir or DATA
    figure_directory = args.output_dir or (RESULTS / 'figures' if args.recompute else FIGURES)
    if args.recompute:
        from vdp_inertia.compute import recompute

        data_directory = RESULTS / 'data'
        recompute(data_directory)

    from vdp_inertia.plot import load_data, set_style, phase_portrait, heatmaps, exit_times

    cycle, fields, cases = load_data(data_directory)
    formats = tuple(dict.fromkeys(args.formats))
    set_style()
    for coordinates in ('original', 'lienard'):
        phase_portrait(cycle, coordinates, figure_directory, formats)
    heatmaps(fields, cases, figure_directory, formats)
    exit_times(cases, figure_directory, formats)
    print(f'Five figures saved as {", ".join(formats)} in {figure_directory.resolve()}')


if __name__ == '__main__':
    main()
