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
    :param graphics_subparsers:
    :return:
    """

    plume_visualization_parser = graphics_subparsers.add_parser('precipitation', help='Precipitation visualization')


def main(*args, **kwargs):
    """
    """
    pass
