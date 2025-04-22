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
from . import flow

from ..defaults import SERTODefaults

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

    flow.configure_subparsers(analysis_subparsers)
    plumes.configure_subparsers(analysis_subparsers)
    precipitation.configure_subparsers(analysis_subparsers)
    wind.configure_subparsers(analysis_subparsers)


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
        plumes.main(parser_args, *args, **kwargs)
    elif parser_args.analysis_command == 'precip':
        precipitation.main(parser_args, *args, **kwargs)
    elif parser_args.analysis_command == 'wind':
        wind.main(parser_args,*args, **kwargs)
    elif parser_args.analysis_command == 'flow':
        flow.main(parser_args, *args, **kwargs)
    else:
        raise ValueError(f"Invalid analysis command: {parser_args.analysis_command}")
