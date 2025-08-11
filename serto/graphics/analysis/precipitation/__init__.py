"""
This module provides the functionality to visualize plumes in SWMM.
"""
# python imports
import argparse

# third-party imports

# local imports
from .precipitation import PrecipitationVisualization

def configure_subparsers(graphics_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the plume command.
    :param graphics_subparsers:
    :return:
    """

    plume_visualization_parser = graphics_subparsers.add_parser(
        'precipitation',
        help='Precipitation visualization'
    )


def main(*args, **kwargs):
    """Main function for the precipitation visualization command.
    """
    pass

