import pandas as pd

from .precipitation import *

# python imports
import argparse
import os.path
import yaml
from typing import List, Callable, Any

# third-party imports
import joblib

# local imports
from . import PrecipitationAnalysis


def configure_subparsers(analysis_subparsers: argparse.ArgumentParser):
    """
    Configure the subparser for the precipitation command.
    :param analysis_subparsers: The subparsers for the analysis command.
    :return: None
    """

    precip_analysis_parser = analysis_subparsers.add_parser('precip', help='Precipitation analysis')

    precip_analysis_subparsers = precip_analysis_parser.add_subparsers(
        title='precipitation analysis subcommands',
        dest='precip_analysis_command',
        help='Additional help for the precipitation analysis subcommands'
    )

    # Event analysis

    precip_event_parser = precip_analysis_subparsers.add_parser('event', help='Precipitation event analysis')

    precip_event_parser.add_argument(
        "--csv",
        help="CSV file containing precipitation data",
        metavar='csv',
        action='store',
        required=True,
    )

    precip_event_parser.add_argument(
        "-t",
        "--inter_event_time",
        help="Inter event time (in hours)",
        metavar='inter_event_time',
        action='store',
        required=True,
    )

    precip_event_parser.add_argument(
        "--lat",
        help="Latitude",
        action='store',
        metavar='lat',
    )

    precip_event_parser.add_argument(
        "--lon",
        help="Longitude",
        action='store',
        metavar='lon',
    )

    precip_event_parser.add_argument(
        "-s",
        "--series",
        help="Precipitation series to use for extracting return periods from NOAA Atlas 14",
        metavar='series',
        action='store',
        choices=["ams", "pds"],
    )

    precip_event_parser.add_argument(
        "-c",
        "--clusters",
        help="Cluster precipitation events based on number of clusters provided",
        metavar='series',
        action='store',
    )

    precip_event_parser.add_argument(
        '--cluster_output',
        help='Output file path for the cluster model used for the clustering events',
        action='store',
        metavar='cluster_output'
    )

    precip_event_parser.add_argument(
        '-o',
        '--output',
        metavar='output',
        help='Output csv file path',
        required=True,
        action='store'
    )

    # IDF curve generation


def main(parser_args: argparse.Namespace, *args, **kwargs):
    """
    Main function for the precipitation analysis module.
    :param parser_args: argparse.Namespace object containing the parsed arguments from the command line interface (CLI)
    :param args: Additional arguments to pass to the main function for the precipitation analysis module.
    :param kwargs: Additional keyword arguments to pass to the main function for the precipitation analysis module.
    """
    if parser_args.precip_analysis_command == 'event':
        extract_events(**vars(parser_args))


def process_args_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for the precipitation analysis module.
    :param func: The function to decorate
    :return: The decorated function
    """

    def wrapper(*args, **kwargs):

        if 'csv' not in kwargs:
            raise ValueError("Missing required argument: csv")
        elif not os.path.exists(kwargs['csv']):
            raise FileNotFoundError(f"File {kwargs['csv']} not found.")
        else:
            ts = pd.read_csv(kwargs['csv'], parse_dates=True, index_col=0)
            kwargs['ts'] = ts

        if 'inter_event_time' not in kwargs and 't' not in kwargs:
            raise ValueError("Missing required argument: inter_event_time")
        else:
            if 't' in kwargs:
                kwargs['inter_event_time'] = kwargs['t']
            kwargs['inter_event_time'] = pd.Timedelta(hours=float(kwargs['inter_event_time']))

        return func(*args, **kwargs)

    return wrapper


@process_args_decorator
def extract_events(
        ts: pd.DataFrame,
        inter_event_time: pd.Timedelta,
        output: str,
        clusters: int = None,
        cluster_output: str = None,
        series: str = None,
        lat: float = None,
        lon: float = None,
        *args,
        **kwargs
) -> None:
    """
    Main function for the precipitation analysis module.
    :param ts: The time series data for the precipitation analysis.
    :param inter_event_time: The inter event time for the analysis.
    :param output: The output file path for the analysis.
    :param clusters: The number of clusters to use for the analysis.
    :param cluster_output: The output file path for the cluster model used for the clustering events.
    :param series: The series to use for the analysis (e.g., AMS or PDS).
    :param lat: The latitude for the analysis.
    :param lon: The longitude for the analysis.
    :param args: Additional arguments to pass to the main function for the precipitation analysis module.
    :param kwargs: Additional keyword arguments to pass to the main function for the precipitation analysis module.
    """

    events = []

    for rainfall_column in ts.columns.values:
        rg_data = ts[[rainfall_column]]

        event = PrecipitationAnalysis.get_events(
            rainfall=rg_data,
            inter_event_time=inter_event_time,
            floor=1.0E-12
        )

        if series is not None:
            event = PrecipitationAnalysis.get_noaa_event_return_periods(
                events=event,
                latitude=lat,
                longitude=lon,
                series=series
            )

        if clusters is not None:
            event['duration_total_hours'] = event['duration'] / pd.Timedelta(hours=1)
            event, cluster_model = PrecipitationAnalysis.cluster_events(
                events=event,
                number_of_clusters=int(clusters),
                cluster_columns=[
                    'duration_total_hours',
                    'precip_peak',
                    'precip_total'
                ]
            )

            if cluster_output is not None:
                # save the cluster model to joblib file
                joblib.dump(cluster_model, cluster_output)

        rename_columns = [(rainfall_column, event_column) for event_column in event.columns]
        rename_columns = dict(zip(event.columns.values, rename_columns))
        event.rename(columns=rename_columns, inplace=True)

        events.append(event)

    event_rain_gages = pd.concat(events, axis='columns')
    event_rain_gages.to_csv(output)


def noaa_atlas14_idf_curve(lat: float, lon: float, series: str, *args, **kwargs):
    """
    Parse the NOAA Atlas 14 configuration file.
    :return: The parsed configuration file.
    """
    # Generate code to hit the NOAA Atlas 14 API to get the IDF curve and return pandas DataFrame
    pass


def idf_generator(ts: pd.DataFrame, lat: float, lon: float, series: str, *args, **kwargs):
    """
    Generate intensity-duration-frequency (IDF) curves for the given latitude, longitude, and series.
    :param ts: The time series data for the precipitation analysis.
    :param lat: The latitude for the analysis.
    :param lon: The longitude for the analysis.
    :param series: The series to use for the analysis (e.g., AMS or PDS).
    :param args: Additional arguments to pass to the main function for the precipitation analysis module.
    :param kwargs: Additional keyword arguments to pass to the main function for the precipitation analysis module.
    """
    pass
