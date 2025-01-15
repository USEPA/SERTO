"""
The serto.graphics module provides the functionality for various graphics tasks.
"""
# python imports
import argparse
import pandas as pd

# third party imports

# local imports
from . import swmm
from .swmm import main as swmm_main, configure_subparsers as configure_swmm_visualization_parsers
from .analysis import main as analysis_main, configure_subparsers as configure_analysis_parsers


def configure_subparsers(sub_parsers: argparse._SubParsersAction):
    """
    Configure the subparser for the graphics command.
    :param sub_parsers:
    :return:
    """
    graphics_parser = sub_parsers.add_parser('graphics', help='Perform specific graphics tasks')
    graphics_subparsers = graphics_parser.add_subparsers(
        title='graphics',
        description='Graphics tasks to execute',
        help='Additional help',
        dest='graphics_command'
    )

    configure_swmm_visualization_parsers(graphics_subparsers)
    configure_analysis_parsers(graphics_subparsers)


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the graphics command.
    :param parser_args: argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments to pass to the main function for the graphics command (if any)
    :param kwargs: Additional keyword arguments to pass to the main function for the graphics command (if any)
    :return: None
    """
    if parser_args.graphics_command == 'swmm' or parser_args.graphics_command == 'swmm_config':
        swmm_main(**vars(parser_args))
