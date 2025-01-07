"""
The serto.ensemble module provides the functionality for ensemble simulations.
"""

# python imports
import argparse


# third party imports

# local imports


def configure_subparsers(sub_parsers: argparse._SubParsersAction):
    """
    Configure the subparser for the ensemble command.
    :param sub_parsers: The subparser object to add the ensemble command to the CLI (command line interface)
    :return: None
    """

    ensemble_parser = sub_parsers.add_parser(
        'ensemble',
        help='Perform ensemble simulations'
    )

    mutually_exclusive_group = ensemble_parser.add_mutually_exclusive_group(required=True)

    mutually_exclusive_group.add_argument(
        '-c',
        '--config',
        type=str,
        help='The configuration file to use for the ensemble simulation',
        action='store',
    )

    mutually_exclusive_group.add_argument(
        '-e',
        '--export',
        type=str,
        help='Export the ensemble configuration example to a yaml/json file',
        action='store',
    )


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the ensemble command.
    :param parser_args:  argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments to pass to the main
    :param kwargs: Additional keyword arguments to pass to the main
    :return: None
    """

    if parser_args.command == 'ensemble':
        run_ensemble(**vars(parser_args))
    else:
        raise ValueError(f"Invalid ensemble command: {parser_args.command}")


def run_ensemble(*args, **kwargs):
    """
    Run the ensemble simulation.
    :param args: Additional arguments to pass to the main
    :param kwargs: Additional keyword arguments to pass to the main
    :return: None
    """
    pass
