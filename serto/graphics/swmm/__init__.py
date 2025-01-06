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
from ..swmm import SpatialSWMM

def configure_subparsers(graphics_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the swmm command.
    :param swmm_visualization_parser:
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


def config_decorator(func: Callable[Any, Any]) -> Callable[Any, Any]:
    """
    Decorator to parse configuration files
    :param func:
    :return:  wrapper function
    """
    def wrapper(*args, **kwargs):

        if 'config' in kwargs:
            # check if yaml or json and parse into kwargs

            config_file = kwargs['config']

            if config_file is None:
                pass
            else:
                if config_file.endswith('.json'):
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    import yaml
                    with open(config_file, 'r') as f:
                        config = yaml.load(f, Loader=yaml.Loader)
                else:
                    raise ValueError(f'Unknown file type for config file: {config_file}')

                kwargs.update(config)
                del kwargs['config']

                if 'inp' in kwargs:
                    inp_file = kwargs['inp']
                    if not os.path.isabs(inp_file):
                        inp_file = os.path.join(os.path.dirname(config_file), inp_file)
                        kwargs['inp'] = inp_file
                    if not os.path.exists(inp_file):
                        raise ValueError(f'Input file does not exist: {inp_file}')

                if 'output' in kwargs:
                    output_file = kwargs['output']
                    if not os.path.isabs(output_file):
                        output_file = os.path.join(os.path.dirname(config_file), output_file)
                        kwargs['output'] = output_file

        return func(*args, **kwargs)


    return wrapper


@config_decorator
def main(inp: str = None, crs: str = None, output:str = None, graphics_command="swmm", *args, **kwargs):
    """
    Main function for the command line interface for the graphics module for SWMM visualization
    :return: None
    """
    if graphics_command == 'swmm':
        swmm = SpatialSWMMViz(inp=inp, crs=crs, output=output)
        swmm.plot()
    elif graphics_command == 'swmm_config':
        swmm_viz = SpatialSWMMViz(
            inp = '<SWMM model file>',
            crs = '<Coordinate reference system (CRS) for the model>',
            output = '<Output file path for the visualization with options [*.png, *.pdf, *.svg, *.html for plotly]>'
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
        raise ValueError(f'Unknown command: {command}')
