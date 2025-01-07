"""
The serto.sensor module provides the functionality for sensor placement optimization.
"""

# python imports
import argparse

# third party imports

# local imports
from .chama import main as chama_main, configure_subparsers as configure_chama_subparsers

def configure_subparsers(sub_parsers: argparse._SubParsersAction):
    """
    Configure the subparser for the chamaoptimizer command.
    :param sub_parser:
    :param sub_parser: The
    :return:
    """

    sensor_placement_parser = sub_parsers.add_parser(
        'sensor',
        help='Sensor placement optimization'
    )

    chama_sensor_placement_subparsers = sensor_placement_parser.add_subparsers(
        title='Sensor placement optimization',
        description='Sensor placement optimization commands to execute',
        help='Additional help',
        dest='sensor_command'
    )

    configure_chama_subparsers(chama_sensor_placement_subparsers)

def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the chamaoptimizer command.
    :param args: Additional arguments to pass to the main function for the chamaoptimizer command (if any)
    :param kwargs: Additional keyword arguments to pass to the main function for the chamaoptimizer command (if any)
    :return:
    """

    if parser_args.sensor_command == 'chama':
        chama_main(**vars(parser_args))
    else:
        raise ValueError(f"Invalid sensor command: {parser_args.sensor_command}")