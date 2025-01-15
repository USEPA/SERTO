"""
sero.analysis.precipitation.PrecipitationAnalysis contains the functionality for precipitation analysis.
"""
# python imports
import os
from typing import Union, List, Tuple, Dict
from datetime import datetime, time
import json
import io

# third party imports

import pandas as pd
import unittest
import requests
import numpy as np
import matplotlib.pyplot as plt
import scipy
from imblearn.pipeline import Pipeline

# local imports
from serto import Defaults


class PrecipitationAnalysis:

    def __init__(self):
        pass

    @staticmethod
    def event_short_name(
            event_date: Union[datetime, pd.Timestamp],
            date_time_format: str = '%m/%d/%Y %H:%M:%S',
            output_date_time_format: str = '%Y%b%dT%H',
            output_prefix: str = '',
            output_suffix: str = '',
    ) -> str:
        """
        This function returns a short name for the event
        :param event_date: The date of the event
        :param date_time_format:  The format of the event date
        :param output_date_time_format: The format of the output date
        :param output_prefix: The prefix to use for the output
        :param output_suffix: The suffix to use for the output
        :return: A string with the event name
        """

        if isinstance(event_date, str):
            event_date = datetime.strptime(event_date, date_time_format)

        event_name = rf'{output_prefix}{event_date.strftime(output_date_time_format)}{output_suffix}'

        return event_name

    @staticmethod
    def post_event_time(events: pd.DataFrame, ts_end: pd.Timestamp) -> pd.DataFrame:
        """
        This function returns a dataframe of the times between the end of an event and the start of a new one
        :param events: A dataframe with the event attributes
        :param ts_end: The end of the timeseries data
        :return: A dataframe with the timeseries data post event
        """
        post_times = events['start'].shift(-1) - events['actual_end']
        return post_times.fillna(ts_end - events.iloc[-1]['end'])

    @staticmethod
    def antecedent_event_time(events: pd.DataFrame, ts_start: pd.Timestamp) -> pd.DataFrame:
        """
        This function returns a dataframe of the antecedent times before an event
        :param events:  A dataframe with the event attributes
        :param ts_start:  The start of the timeseries data
        :return: A dataframe with the timeseries data antecedent to the event
        """
        antecedent_times = events['start'] - events['actual_end'].shift(1)
        return antecedent_times.fillna(events.iloc[0]['start'] - ts_start)

    @staticmethod
    def get_noaa_event_return_periods(
            latitude: float,
            longitude: float,
            events: pd.DataFrame,
            units: str = 'english',
            series: str = 'pds'
    ) -> pd.DataFrame:
        """
        Retrieves interpolated NOAA Atlas 14 precipitation frequency estimates for a location of interest
        :param latitude: Latitude of the location of interest
        :param longitude: Longitude of the location of interest
        :param events: Pandas dataframe of events
        :param units: Units of to use for the event attributes. Valid values are 'english' and 'metric'
        :param series: Series to use for the event attributes. Valid values are Partial Duration 'pds' and
        Annual Maximum Series 'ams'
        :return: A dataframe with the event attributes
        """

        query_params = {
            'lat': latitude,
            'lon': longitude,
            'data': 'depth',
            'units': units,
            'series': series,
        }

        response = requests.get(
            url=f'{Defaults.NOAA_PFDS_REST_SERVER_URL}fe_text_mean.csv',
            params=query_params, verify=False
        )

        content = response.content.decode('utf-8')
        results = pd.read_csv(io.StringIO(content), skiprows=13, skipfooter=3, index_col=0, engine='python')

        durations = []

        for duration in results.index:
            duration = duration.replace(':', '')
            duration = duration.replace('-', '')
            if 'min' in duration:
                duration = float(duration.replace('min', ''))
            elif 'hr' in duration:
                duration = float(duration.replace('hr', '')) * 60.0
            elif 'day' in duration:
                duration = float(duration.replace('day', '')) * 60.0 * 24.0

            durations.append(duration)

        durations = np.array(durations)
        results.index = durations
        result_columns = results.columns.to_list()
        results['0'] = 0.0
        results = results[['0'] + result_columns]
        return_periods = np.array([float(c.replace("\'1/", "")) for c in results.columns])

        interp = scipy.interpolate.RegularGridInterpolator(
            points=(durations, return_periods),
            bounds_error=False,
            values=results.values
        )

        events_cp = events.copy()
        events_cp['return_period'] = None

        for event_index, event_row in events_cp.iterrows():
            event_duration = event_row['duration'].total_seconds() / 60.0
            event_precip_total = event_row['precip_total']
            event_durations = np.array([event_duration])
            precip_totals = interp((event_durations, return_periods))
            return_period = np.interp(x=event_precip_total, xp=precip_totals, fp=return_periods)
            events_cp.loc[event_index, 'return_period'] = return_period

        return events_cp

    @staticmethod
    def get_events(
            rainfall: pd.DataFrame,
            inter_event_time: pd.Timedelta = pd.Timedelta('24 hour'),
            floor: Union[float, Dict[str, float]] = 0.01,
            flow_or_depth: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        This function returns a dataframe of events from a rainfall dataframe. An event is defined as a period of time
        :param rainfall: A dataframe with a datetime index and multiple columns representing incremental rainfall
        accumulations in the area of interest
        :param inter_event_time:
        :param floor: Threshold for the minimum rainfall depth or flow for an event to be considered
        :param flow_or_depth:
        :return: A data frame with the event attributes
        """

        data = rainfall.copy()

        if flow_or_depth is not None:
            data = pd.concat(objs=[data, flow_or_depth], axis='columns')

        if isinstance(floor, dict):
            floor = pd.Series(floor)
            floor = floor.reindex(data.columns)
            floor = floor.fillna(0.0)
        elif isinstance(floor, (int, float)):
            floor = pd.Series([floor] * len(data.columns), index=data.columns)

        data = data - floor
        rolling_sum = data.rolling(inter_event_time).sum()
        rolling_sum_combined = rolling_sum.sum(axis='columns')
        max_of_fields = rolling_sum_combined.max(axis="rows")

        normalized_rolling_sum = rolling_sum_combined / max_of_fields
        normalized_rolling_sum[normalized_rolling_sum.index[0]] = 0
        normalized_rolling_sum[normalized_rolling_sum.index[-1]] = 0
        normalized_rolling_sum_index = (normalized_rolling_sum <= 0.0).astype(int)
        normalized_rolling_sum_index_diff = normalized_rolling_sum_index.diff()

        events = pd.DataFrame(
            {
                "start": normalized_rolling_sum_index_diff[normalized_rolling_sum_index_diff == -1].index,
                "end": normalized_rolling_sum_index_diff[normalized_rolling_sum_index_diff == 1].index,
            }
        )

        ts_start, ts_end = data.index[0], data.index[-1]

        events['name'] = events.apply(
            lambda row: PrecipitationAnalysis.event_short_name(row['start'], output_prefix='', output_suffix=''),
            axis='columns'
        )

        events = events[['name', 'start', 'end']]

        def get_actual_end(row, ts):
            sub_ts = ts[row['start']:row['end']]

            return sub_ts.loc[sub_ts[sub_ts.columns[0]] > 0].dropna().index[-1]

        events['actual_end'] = events.apply(func=get_actual_end, axis='columns', args=(rainfall,))

        events['duration'] = events['actual_end'] - events['start']
        events['post_event_time'] = PrecipitationAnalysis.post_event_time(events, ts_end)
        events['antecedent_event_time'] = PrecipitationAnalysis.antecedent_event_time(events, ts_start)

        def get_rainfall_peak_intensity(row, ts):
            return ts[row['start']:row['end']].max()

        def get_rainfall_sum(row, ts):
            return ts[row['start']:row['end']].sum()

        events['precip_peak'] = events.apply(func=get_rainfall_peak_intensity, axis='columns', args=(rainfall,))
        events['precip_total'] = events.apply(func=get_rainfall_sum, axis='columns', args=(rainfall,))

        return events

    @staticmethod
    def write_swmm_rainfall_file_from_events(
            rainfall: pd.DataFrame,
            events: pd.DataFrame,
            output_file_path: str,
            output_file_prefix: str = '',
            output_file_suffix: str = '',
    ) -> None:
        """
        This function writes a SWMM rainfall file for each event
        :param rainfall: A dataframe with the rainfall data
        :param events: A dataframe with the event attributes
        :param output_file_path: Path to write the output file
        :param output_file_prefix: Prefix to use for the output file name
        :param output_file_suffix: Suffix to use for the output file name
        :return:
        """
        for _, event_row in events.iterrows():
            event_name = f'{output_file_prefix}{event_row["name"]}{output_file_suffix}'
            filename = os.path.join(output_file_path, f'{event_name}.dat')
            ts = rainfall[event_row['start']:event_row['end']]
            PrecipitationAnalysis.write_swmm_rainfall_file(ts, event_name, filename)

    @staticmethod
    def write_swmm_rainfall_file(
            rainfall: pd.DataFrame,
            event_name: str,
            output_file: str
    ) -> None:
        """
        This function writes a SWMM rainfall file
        :param rainfall: A dataframe with the rainfall data
        :param event_name: The name of the event
        :param output_file: Path to write the output file
        :return:
        """
        with open(output_file, 'w') as f:
            f.write(f';;Event {event_name}\n')
            for index, row in rainfall.iterrows():
                f.write(f'{index.strftime("%m/%d/%Y %H:%M:%S")}\t{row.values[0]}\n')

    @staticmethod
    def cluster_events(events: pd.DataFrame, cluster_columns: List[str], number_of_clusters: int) -> Tuple[
        pd.DataFrame, Pipeline]:
        """
        This function clusters events based on the specified columns
        :param events: A dataframe with the event attributes
        :param cluster_columns: A list of columns to use for clustering
        :param number_of_clusters: The number of clusters to create
        :return: A tuple with a dataframe with the event attributes and cluster labels
        and the clustering model
        """
        from sklearn.cluster import KMeans
        from imblearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler

        model = Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                ('kmeans', KMeans(n_clusters=number_of_clusters))
            ]
        )

        cluster_data = events[cluster_columns].copy()
        model.fit(cluster_data)
        cluster_labels = model.predict(cluster_data)
        events['cluster'] = cluster_labels

        return events, model

    @staticmethod
    def sample_rainfall_events(
            events: pd.DataFrame,
            probability_column: str,
            number_of_samples: int,
            with_replacement: bool = False,
            random_state: int = 42
    ) -> pd.DataFrame:
        """
        This function samples rainfall events based on the specified probability column
        :param events: A dataframe with the event attributes
        :param probability_column: The column with the probability values
        :param number_of_samples: The number of samples to take
        :param with_replacement: Whether to sample with replacement
        :param random_state: The random state to use for sampling
        :return: A dataframe with the sampled events
        """
        return events.sample(
            n=number_of_samples,
            weights=probability_column,
            replace=with_replacement,
            random_state=random_state
        )
