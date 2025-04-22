"""
The serto.analysis module provides the functionality for various analysis tasks.
"""

# python imports
import argparse

# third party imports

# local imports
from . import plumes
from . import precipitation
from . import wind

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
    plumes.configure_subparsers(analysis_subparsers)
    precipitation.configure_subparsers(analysis_subparsers)
    wind.configure_subparsers(analysis_subparsers)


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the analysis command.
    :param parser_args: argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments
    :param kwargs: Additional keyword arguments
    :return: None
    """
    if parser_args.analysis_command == 'plumes':
        plumes.main(**vars(parser_args), *args, **kwargs)
    elif parser_args.analysis_command == 'precipitation':
        precipitation.main(**vars(parser_args) , *args, **kwargs)
    elif parser_args.analysis_command == 'wind':
        wind.main(**vars(parser_args), *args, **kwargs)
    else:
        raise ValueError(f"Invalid analysis command: {parser_args.analysis_command}")
