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
from .plumes import GaussianPlume, GaussianPlumeVisualization

def configure_subparsers(graphics_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the plume command.
    :param plume_visualization_parser:
    :return:
    """

    plume_visualization_parser = graphics_subparsers.add_parser('plume', help='Plume visualization')

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
        "--crs",
        help="Coordinate reference system",
        action="store",
        required=True,
    )

    plume_visualization_parser.add_argument(
        "--value",
        help="Contaminant initial value",
        action="store",
        required=True,
    )

    plume_visualization_parser.add_argument(
        "--ptype",
        help="Plume type",
        action="store",
        choices=["EMPIRICAL", "PHYSICS_BASED"],
    )

    plume_visualization_parser.add_argument(
        "--spres",
        help="Spatial resolution of the plume",
        action="store",
        default=50,
    )

    plume_visualization_parser.add_argument(
        "--pname",
        help="Name of pollutant",
        action="store",
        default="Cesium",
    )

    plume_visualization_parser.add_argument(
        "--units",
        help="Units of the pollutant",
        action="store",
        default="Curies",
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