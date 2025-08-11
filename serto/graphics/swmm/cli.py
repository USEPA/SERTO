"""
SWMM visualization module for the Serto graphics package
"""

# python imports
import argparse
import os.path
import yaml
from typing import List, Callable, Any

# third-party imports

# local imports
from .spatialswmm import *
from ...swmm import SpatialSWMM


def configure_subparsers(graphics_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the swmm command.
    :param graphics_subparsers: The subparsers for the graphics command.
    :return:
    """
    swmm_visualization_parser = graphics_subparsers.add_parser('swmm', help='SWMM IO visualization')
    mutually_exclusive_group = swmm_visualization_parser.add_mutually_exclusive_group(required=True)

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

    swmm_visualization_parser.add_argument(
        '-p',
        '--crs',
        default='EPSG:4326',
        help='Coordinate reference system (CRS) for the model',
        action='store'
    )

    swmm_visualization_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path for the visualization with options ['*.png', '*.pdf', '*.svg', '*.html' for plotly]",
        action='store'
    )

    swmm_visualization_config_parser = graphics_subparsers.add_parser(
        'swmm_config',
        help='Export SWMM model vizualization configuration configuration template to yaml or json',
    )

    swmm_visualization_config_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file example configuration file ['*.json', '*.yaml', '*.yml']",
        action='store'
    )


def main(inp: str = None, crs: str = None, output: str = None, graphics_command="swmm", *args, **kwargs):
    """
    Main function for the command line interface for the graphics module for SWMM visualization
    :return: None
    """
    if graphics_command == 'swmm':
        swmm = SpatialSWMMVisualization(inp=inp, crs=crs, output=output)
        swmm.plot()
    elif graphics_command == 'swmm_config':
        swmm_viz = SpatialSWMMVisualization(
            inp='<SWMM model file>',
            crs='<Coordinate reference system (CRS) for the model>',
            output='<Output file path for the visualization with options [*.png, *.pdf, *.svg, *.html for plotly]>'
        )

        # Check if the output file is either json or yaml and export formatted dict to file
        if output.endswith('.json'):
            import json
            with open(output, 'w') as f:
                json.dump(swmm_viz.to_dict(), f)
        elif output.endswith('.yaml') or output.endswith('.yml'):
            import yaml
            with open(output, 'w') as f:
                yaml.dump(swmm_viz.to_dict(), f)
        else:
            raise ValueError(f'Unknown file type for config file: {output}')
    else:
        raise ValueError(f'Unknown command: {graphics_command}')
