"""
This module contains the classes and functions for the plume visualization.
"""
from .gaussianplume import GaussianPlume

# python imports
import argparse
import os.path
import yaml
from typing import List, Callable, Any

# third-party imports

# local imports

def configure_subparsers(analysis_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the plume command.
    :param plume_visualization_parser:
    :return:
    """

    plume_visualization_parser = analysis_subparsers.add_parser('analysis', help='Plume visualization')
    plume_visualization_parser.add_argument(
        "config",
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