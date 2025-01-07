"""
The serto.analysis module provides the functionality for various analysis tasks.
"""

# python imports
import argparse

# third party imports

# local imports
from .plumes import main as plumes_main, configure_subparsers as configure_plumes_subparsers
from .precipitation import main as precipitation_main, configure_subparsers as configure_precipitation_subparsers
from .wind import main as wind_main, configure_subparsers as configure_wind_subparsers


def configure_subparsers(sub_parsers: argparse._SubParsersAction):
    """
    Configure the subparser for the analysis command.
    :param sub_parsers:
    :return: None
    """
    analysis_parser = sub_parsers.add_parser('analysis', help='Perform specific analysis tasks')
    analysis_subparsers = analysis_parser.add_subparsers(
        title='analysis',
        description='Analysis tasks to execute',
        help='Additional help',
        dest='analysis_command'
    )

    # add subparsers here
    configure_plumes_subparsers(analysis_subparsers)
    configure_precipitation_subparsers(analysis_subparsers)
    configure_wind_subparsers(analysis_subparsers)


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the analysis command.
    :param parser_args: argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments
    :param kwargs: Additional keyword arguments
    :return: None
    """
    if parser_args.analysis_command == 'plumes':
        plumes_main(**vars(parser_args))
    elif parser_args.analysis_command == 'precipitation':
        precipitation_main(**vars(parser_args))
    elif parser_args.analysis_command == 'wind':
        wind_main(**vars(parser_args))
    else:
        raise ValueError(f"Invalid analysis command: {parser_args.analysis_command}")
