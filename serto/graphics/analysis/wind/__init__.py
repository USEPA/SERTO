"""
    This module contains the wind visualization functionality for the Serto project.
"""
# python imports
import argparse

# third-party imports

# local imports
from .wind import *

def configure_subparsers(analysis_subparsers: argparse._SubParsersAction):
    """
    Configure the subparser for the plume command.
    :param analysis_subparsers: argparse.ArgumentParser object
    :return:
    """

    wind_data_visualization_parser = analysis_subparsers.add_parser('wind', help='Wind visualization')

    mutually_exclusive_group = wind_data_visualization_parser.add_mutually_exclusive_group(required=True)
    mutually_exclusive_group.add_argument(
        "-i",
        "--inp",
        help="SWMM input file",
        action="store",
    )

    mutually_exclusive_group.add_argument(
        "-c",
        "--config",
        help="Configuration file for the visualization",
        action='store',
    )

    wind_data_visualization_parser.add_argument(
        '-o',
        '--output',
        help='Output file path',
        required=True,
        action='store'
    )


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for wind visualization.
    """
    pass