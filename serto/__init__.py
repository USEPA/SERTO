VERSION_INFO = (0, 0, 0, "dev1")
__version__ = ".".join(map(str, VERSION_INFO))
__author__ = "Caleb Buahin and Anne Mikelonis"
__copyright__ = "Copyright (c) 2024"
__credits__ = ["Caleb Buahin and Anne Mikelonis"]
__license__ = "MIT"
__maintainer__ = "Caleb Buahin and Anne Mikelonis"
__email__ = "buahin.caleb@epa.gov"
__status__ = "Development"

# python imports
import argparse

# third party imports

# local imports
from .defaults import SERTODefaults, IDictable, SERTONumpyEncoder
from . import swmm
from . import analysis
from . import ensemble
from . import graphics
from . import moo
from . import sensor



def configured_parser() -> argparse.ArgumentParser:
    """
    Configure the parser for the command line interface
    :return: (argparse.ArgumentParser) The configured parser
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
    analysis.configure_subparsers(subparsers)
    graphics.configure_subparsers(subparsers)
    sensor.configure_subparsers(subparsers)
    moo.configure_subparsers(subparsers)
    ensemble.configure_subparsers(subparsers)

    return parser

def parse_args(parser, args):
    """
    Parse the command line arguments
    :param parser: (argparse.ArgumentParser) The configured parser for the command line interface
    :param args: (List[str]) The command line arguments
    :return: (argparse.Namespace) The parsed arguments
    """

    args = parser.parse_args(args)

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


def main():
    """
    Main function for the command line interface
    :return:
    """
    parser = configured_parser()
    args = parser.parse_args()
    parse_args(parser, args)


if __name__ == '__main__':
    main()
