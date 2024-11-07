import argparse
from . import __version__


def main():
    """
    Main function for the command line interface
    :return:
    """

    parser = argparse.ArgumentParser(
        prog='serto',
        description='The Stormwater Emergency Response Tool & Optimizer (SERTO)',
        epilog='Developed by the US EPA Office of Research and Development',
    )

    parser.add_argument(
        "-v","--verbose",
        help="Increase output verbosity",
        action="store_true",
    )

    # add version argument
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )

    subparsers = parser.add_subparsers(dest='command')

    sensor_placement_parser = subparsers.add_parser('sp', help='Sensor placement optimization')
    sensor_placement_parser.add_argument(
        "-c", "--config",
        type=str,
        help='Path to the configuration file'
    )

    analysis_parser = subparsers.add_parser('analysis', help='Perform specific analysis')
    analysis_parser.add_argument(
        "-c", "--config",
        type=str,
        help='Path to the configuration file'
    )

    multiobjective_optimization_parser = subparsers.add_parser(
        name='moo',
        help='Perform multi-objective optimization'
    )
    multiobjective_optimization_parser.add_argument(
        "-c", "--config",
        type=str,
        help='Path to the configuration file'
    )

    ensemble_parser = subparsers.add_parser(
        name='ensemble',
        help='Create an ensemble of models and run them locally, an HPC, or in the cloud'
    )
    ensemble_parser.add_argument(
        "-c", "--config",
        type=str,
        help='Path to the configuration file'
    )

    graphics_parser = subparsers.add_parser('graphics', help='Create graphics for the results')
    graphics_parser.add_argument(
        "-c", "--config",
        type=str,
        help='Path to the configuration file'
    )

    args = parser.parse_args()

    # if args.command == 'run':
    #     if args.opti_type == 'sensor_placement_chama.yml':
    #         from swmmoptimizer import chama_optimization
    #         chama_optimization.run(args.config, use_existing=args.use_existing)


if __name__ == '__main__':
    main()