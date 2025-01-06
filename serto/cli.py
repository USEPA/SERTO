# python imports
import argparse

# third party imports


# local imports
from . import __version__
from . import analysis
from . import ensemble
from . import graphics
from . import moo
from . import sensor

def main():
    """
    Main function for the command line interface
    :return:
    """

    parser = argparse.ArgumentParser(
        prog='serto',
        description='The Stormwater Emergency Response Tool & Optimizer (SERTO)\n'
                    'for emergency response and optimization applications in stormwater systems',
        epilog='Developed by the US EPA Office of Research and Development',
    )

    parser.add_argument(
        "-v", "--verbose",
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

    # Configure the subparsers
    graphics.configure_subparsers(subparsers)
    analysis.configure_subparser(subparsers)


    args = parser.parse_args()

    if args.command == 'sp':
        from .sensor import main as sp_main
        sp_main(args)
    elif args.command == 'analysis':
        from .analysis import main as analysis_main
        analysis_main(args)
    elif args.command == 'moo':
        from .moo import main as moo_main
        moo_main(args)
    elif args.command == 'ensemble':
        from .ensemble import main as ensemble_main
        ensemble_main(args)
    elif args.command == 'graphics':
        from .graphics import main as graphics_main
        graphics_main(args)


if __name__ == '__main__':
    main()
