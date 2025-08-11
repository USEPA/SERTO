"""
This module contains the classes and functions for platypus multi-objective optimization.
"""
# python imports
import argparse
import os.path
import yaml
from typing import List, Callable, Any


# third-party imports

# local imports


def configure_subparsers(analysis_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the platypus command.
    :param analysis_subparsers: The subparsers for the analysis command.
    :return: None
    """

    platypus_parser = analysis_subparsers.add_parser('platypus', help='Platypus multi-objective optimization')
    platypus_parser.add_argument(
        "config",
        help="Configuration file for platypus optimization",
        action='store',
    )

    platypus_parser.add_argument(
        '-o',
        '--output',
        help='Output file path',
        required=True,
        action='store'
    )

    platypus_parser.add_argument(
        '-n',
        '--nruns',
        help='Number of optimization runs',
        required=False,
        action='store',
        default=1
    )

    platypus_parser.add_argument(
        '-p',
        '--population',
        help='Population size',
        required=False,
        action='store',
        default=100
    )

    platypus_parser.add_argument(
        '-g',
        '--generations',
        help='Number of generations',
        required=False,
        action='store',
        default=100
    )

    platypus_parser.add_argument(
        '-s',
        '--seed',
        help='Random seed',
        required=False,
        action='store',
        default=None
    )


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the platypus optimization module.
    :param parser_args: argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments to pass to the main function for the platypus optimization module.
    :param kwargs:
    :return:
    """
    if parser_args.moo_command == 'platypus':
        platypus_main(**vars(parser_args))


def platypus_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for the platypus optimization module.
    :param func: The function to decorate
    :return: The decorated function
    """

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@platypus_decorator
def platypus_main(*args, **kwargs):
    """
    Main function for the platypus optimization module.
    :param args: Additional arguments to pass to the main
    function for the platypus optimization module.
    :param kwargs: Additional keyword arguments to pass to the main
    function for the platypus optimization module.
    """
    pass
