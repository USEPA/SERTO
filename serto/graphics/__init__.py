"""
The serto.graphics module provides the functionality for various graphics tasks.
"""
# python imports
import argparse
import pandas as pd


# third party imports

# local imports
from . import swmm
from .swmm import main as swmm_main, configure_subparsers as configure_swmm_visualization_parser
# from .analysis import main as analysis_main, configure_subparsers as configure_analysis_parser


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

    configure_swmm_visualization_parser(graphics_subparsers)
    # configure_analysis_parser(graphics_subparsers)

def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    """
    if parser_args.graphics_command == 'swmm' or parser_args.graphics_command == 'swmm_config':
        swmm_main(**vars(parser_args))
