"""
The serto.analysis module provides the functionality for various analysis tasks.
"""

# python imports
import argparse
import pandas as pd

# third party imports

# local imports
from ..defaults import SERTODefaults
from .plumes import main as plumes_main, configure_subparsers as configure_plumes_subparsers
from .precipitation import main as precipitation_main, configure_subparsers as configure_precipitation_subparsers
from .wind import main as wind_main, configure_subparsers as configure_wind_subparsers
from .flow import main as flow_main, configure_subparsers as configure_flow_subparsers


def configure_subparsers(sub_parsers: argparse._SubParsersAction):
    """
    Configure the subparser for the analysis command.
    :param sub_parsers: (argparse._SubParsersAction) The subparsers for the analysis command.
    :return:
    """
    analysis_parser = sub_parsers.add_parser('analysis', help='Perform specific analysis tasks')
    analysis_subparsers = analysis_parser.add_subparsers(
        title='analysis',
        description='Analysis tasks to execute',
        help='Additional help',
        dest='analysis_command'
    )

    configure_flow_subparsers(analysis_subparsers)
    configure_plumes_subparsers(analysis_subparsers)
    configure_precipitation_subparsers(analysis_subparsers)
    configure_wind_subparsers(analysis_subparsers)


@SERTODefaults.extra_args()
def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the analysis commands.
    :param parser_args: (argparse.Namespace) An object containing the parsed arguments
    from the command line interface (CLI)
    :param args: Additional arguments to pass to the main function for the analysis command (if any)
    :param kwargs: Additional keyword arguments to pass to the main function for the analysis command (if any)
    """
    if parser_args.analysis_command == 'plumes':
        plumes_main(parser_args, *args, **kwargs)
    elif parser_args.analysis_command == 'precip':
        precipitation_main(parser_args, *args, **kwargs)
    elif parser_args.analysis_command == 'wind':
        wind_main(parser_args,*args, **kwargs)
    elif parser_args.analysis_command == 'flow':
        flow_main(parser_args, *args, **kwargs)
    else:
        raise ValueError(f"Invalid analysis command: {parser_args.analysis_command}")
