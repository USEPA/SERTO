# python imports
import argparse
import os.path
import yaml
from typing import List, Callable, Any
import pathlib

# third-party imports

# local imports
from . import PlumeEventMatrix
from ... import SERTODefaults


def configure_subparsers(analysis_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the plume command.
    :param analysis_subparsers: The subparsers for the analysis command.
    :return: None
    """

    plume_analysis_parser = analysis_subparsers.add_parser('plumes', help='Plume analysis')

    plume_analysis_parser.add_argument(
        metavar='swmm_input_file',
        help='SWMM input file path for the plume analysis',
        dest='swmm_input_file',
        type=pathlib.Path,
        action='store'
    )

    plume_analysis_parser.add_argument(
        metavar='plume_type',
        help='The plume formulation type to use. (choices: %(choices)s)',
        choices=['EMPIRICAL', 'PHYSICS_BASED'],
        dest='plume_type',
        type=str,
        action='store'
    )

    plume_analysis_parser.add_argument(
        metavar='contaminant_loading',
        help='The initial source contaminant loading amounts as total amount',
        nargs='+',
        type=float,
        dest='contaminant_loading',
        action='store'
    )

    plume_analysis_parser.add_argument(
        "-p",
        "--crs",
        metavar='coordinate_reference_system',
        help="Coordinate reference system (CRS) for the model. (default: %(default)s)",
        action='store',
        default='EPSG:4326',
        type=str,
    )

    plume_analysis_parser.add_argument(
        "-n",
        "--cname",
        metavar='contaminant_name',
        help="The name of the type of contaminant. (default: %(default)s)",
        action='store',
        default='Cesium',
        type=str,
    )

    plume_analysis_parser.add_argument(
        "-u",
        "--units",
        metavar='units',
        help="The units of the contaminant. (default: %(default)s)",
        action='store',
        default='Curies',
        type=str,
    )

    plume_analysis_parser.add_argument(
        "-w",
        "--wind",
        metavar='wind',
        help="Wind speed and direction for the plume analysis (m/s, degrees)",
        nargs='+',
        type=SERTODefaults.coordinates,
        action='store',
    )

    plume_analysis_parser.add_argument(
        "-s",
        "--wind_scenario",
        metavar='wind_scenario',
        help="Wind scenarios analysis output from wind analysis module provided as a yml file",
        action='store',
        type=pathlib.Path,
    )

    plume_analysis_parser.add_argument(
        "-d",
        "--stddev",
        metavar='stddev',
        help="Standard deviation for the plume analysis. (type: downwind, crosswind). [EMPIRICAL formulation only]",
        action='store',
        type=SERTODefaults.coordinates,
        nargs='+',
    )

    plume_analysis_parser.add_argument(
        "-e",
        "--release_element_types",
        metavar='release_element_types',
        help="Element types whose locations are used for the release of the plume. (default: all %(default)s)",
        action='store',
        default=['JUNCTIONS'],
        nargs='+',
        choices=['JUNCTIONS', 'OUTFALLS', 'STORAGES', 'DIVIDERS'],
        type=str,
    )

    plume_analysis_parser.add_argument(
        "-l",
        "--release_locations",
        metavar='release_locations',
        help="Locations for the release of the plume analysis. (type: name, x, y)",
        action='store',
        type=SERTODefaults.named_coordinates,
        nargs='+',
    )

    plume_analysis_parser.add_argument(
        "-g",
        "--release_grid",
        metavar='release_grid',
        help="Grid for the release of the plume analysis. (default: %(default)s)",
        default=[100, 100],
        action='store',
        type=SERTODefaults.coordinates,
        nargs='+',
    )

    plume_analysis_parser.add_argument(
        "-c",
        "--config",
        metavar='config',
        help="Configuration file containing additional parameters for the plume analysis",
        action='store',
        type=pathlib.Path,
    )

    plume_analysis_parser.add_argument(
        '-o',
        '--output',
        metavar='output',
        help="File path to yaml or json output file for the plume analysis",
        required=True,
        action='store',
        type=pathlib.Path
    )

    plume_analysis_parser.add_argument(
        '--shp',
        metavar='Shapefile',
        help="File path to output shapefile for the plume analysis",
        required=False,
        action='store',
        type=pathlib.Path
    )


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the analysis command.
    :param parser_args: argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments to pass to the main function for the analysis command (if any)
    :param kwargs: Additional keyword arguments to pass to the main function for the analysis command (if any)
    :return: None
    """
    if parser_args.analysis_command == 'plumes':
        PlumeEventMatrix.generate_plumes(*args, {**vars(parser_args), **kwargs})
    else:
        raise ValueError(f"Invalid analysis command: {parser_args.analysis_command}")


