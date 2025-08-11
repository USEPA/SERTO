"""
The serto.moo.cli provides the command line interface (CLI) for multi-objective optimization (MOO) algorithms.
"""
# python imports
import argparse

# third party imports

# local imports
from .platypus import main as platypus_main, configure_subparsers as configure_platypus_subparsers


def configure_subparsers(sub_parsers: argparse._SubParsersAction):
    """
    Configure the subparser for the moo command.
    :param sub_parsers:  The subparser object to add the moo command to the CLI
    :return: None
    """
    moo_parser = sub_parsers.add_parser(
        'moo',
        help='Perform multi-objective optimization'
    )

    moo_subparsers = moo_parser.add_subparsers(
        title='moo',
        description='Multi-objective optimization algorithms to execute',
        help='Additional help',
        dest='moo_command'
    )

    # configure the subparsers for the moo command
    configure_platypus_subparsers(moo_subparsers)


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the moo command.
    :param parser_args:  argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments to pass to the main
    :param kwargs: Additional keyword arguments to pass to the main
    :return: None
    """
    if parser_args.moo_command == 'platypus':
        platypus_main(**vars(parser_args))
    else:
        raise ValueError(f"Invalid moo command: {parser_args.moo_command}")
