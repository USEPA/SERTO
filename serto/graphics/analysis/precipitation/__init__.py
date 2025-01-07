"""
This module provides the functionality to visualize plumes in SWMM.
"""
# python imports
import argparse
import os.path
import yaml
from typing import List, Callable, Any

# third-party imports

# local imports

def configure_subparsers(graphics_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the plume command.
    :param plume_visualization_parser:
    :return:
    """

    plume_visualization_parser = graphics_subparsers.add_parser('precipitation', help='Precipitation visualization')

    mutually_exclusive_group = plume_visualization_parser.add_mutually_exclusive_group(required=True)
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

    plume_visualization_parser.add_argument(
        '-o',
        '--output',
        help='Output file path',
        required=True,
        action='store'
    )


def main(*args, **kwargs):
    """
    """
    pass