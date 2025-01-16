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
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.pyplot as plt
import scipy
from imblearn.pipeline import Pipeline

# local imports
from ... import SERTODefaults, IDictable


class PrecipAnalysisData(IDictable):
    """
    This class contains the wind data to be used for wind analysis
    """
    class SamplingApproach:
        """
        This class defines the
        """
        RANDOM = 0
        GAUSSIAN_MIXTURE = 1
        BAYESIAN_GAUSSIAN_MIXTURE = 2

    def __init__(
            self,
            data: Union[str,pd.DataFrame] = None,
            precipitation_col: str = None,
            inter_event_time: float = 24.0,
            latitude: float = 39.049774180652236,
            longitude: float = -84.66127045790225,
            atlas_14_series: str = 'ams',
            num_event_clusters: int = 6,
            sampling_approach: int = SamplingApproach.RANDOM
    ):
        """
        Constructor for the precipitation analysis data object.
        :param data: The wind precipitation to use for the analysis (either a path to a CSV file or a pandas DataFrame)
        :param precipitation_col: Precipitation column name if the data is a pandas DataFrame
        :param inter_event_time: The time between events in hours (default is 24 hours)
        :param latitude: The latitude of the location of interest
        :param longitude: The longitude of the location of interest
        :param atlas_14_series: The NOAA Atlas 14 series to use for the event attributes (default is 'ams'). Options are
        'ams' for Annual Maximum Series and 'pds' for Partial Duration Series
        :param num_event_clusters: The number of clusters to use for event clustering
        :param sampling_approach: The sampling approach to use for event sampling
        """
        self._data = data
        self._precipitation_col = precipitation_col
        self.inter_event_time = inter_event_time
        self.latitude = latitude
        self.longitude = longitude
        self.atlas_14_series = atlas_14_series
        self.sampling_approach = sampling_approach
        self.num_event_clusters = num_event_clusters
        self._model = None

        if isinstance(data, pd.DataFrame):
            self._model_data = data[[precipitation_col]]
        elif isinstance(data, str):
            self._model_data = pd.read_csv(data, index_col=0, parse_dates=True)[[precipitation_col]]
        else:
            raise ValueError('Invalid data type')

        self._events = self._get_events()
    @property
    def data(self) -> Union[NDArray, pd.DataFrame]:
        """
        Get the wind data for the analysis object
        :return:
        """
        return self._model_data

    def _get_events(self) -> pd.DataFrame:
        """
        This function returns a dataframe of events from a rainfall dataframe. An event is defined as a period of time
        :return:
        """

        events = PrecipitationAnalysis.get_events(
            rainfall=self._model_data,
            inter_event_time=pd.Timedelta(hours=self.inter_event_time),
            floor=0.01
        )

        events = PrecipitationAnalysis.get_noaa_event_return_periods(
            events=events,
            latitude=self.latitude,
            longitude=self.longitude,
            series=self.atlas_14_series
        )

        events, self._model = PrecipitationAnalysis.cluster_events(
            events=events,
            cluster_columns=['precip_total', 'precip_peak', 'duration_hours'],
            number_of_clusters=self.num_event_clusters,
            clustering_model='bayesian_gaussian_mixture'
        )

        return events

    def sample_events(self, number_of_samples: int) -> pd.DataFrame:
        """
        This function samples rainfall events based on the specified probability column
        :param number_of_samples: The number of samples to take
        :return: A dataframe with the sampled events
        """
        events = self._get_events()

        if self.sampling_approach == self.SamplingApproach.RANDOM:
            index = np.random.choice(events.index, number_of_samples, replace=False, p=1.0/events['return_period'])
            return events.loc[index]

        elif (self.sampling_approach == self.SamplingApproach.GAUSSIAN_MIXTURE or
              self.sampling_approach == self.SamplingApproach.BAYESIAN_GAUSSIAN_MIXTURE):
            probabilities = self._model.predict_proba(events[['precip_total', 'precip_peak', 'duration_hours']])
            index = np.random.choice(events.index, number_of_samples, replace=False, p=probabilities)
            return events.loc[index]
        else:
            raise ValueError('Invalid sampling approach')

    def to_dict(self, base_directory: str = None) -> dict:
        """
        Convert the object to a dictionary
        :param base_directory: The base directory for relative paths
        :return: The dictionary
        """
        if isinstance(self._data, str):
            data_dict = {
                'data': self._data,
                'precipitation_col': self._precipitation_col,
                'inter_event_time': self.inter_event_time,
                'latitude': self.latitude,
                'longitude': self.longitude,
                'atlas_14_series': self.atlas_14_series,
                'sampling_approach': self.sampling_approach,
                'num_event_clusters': self.num_event_clusters

            }
        else:
            data_dict = {
                'data': self._data.to_dict(),
                'precipitation_col': self._precipitation_col,
                'inter_event_time': self.inter_event_time,
                'latitude': self.latitude,
                'longitude': self.longitude,
                'atlas_14_series': self.atlas_14_series,
                'sampling_approach': self.sampling_approach,
                'num_event_clusters': self.num_event_clusters
            }


        return data_dict

    @classmethod
    def from_dict(cls, data: dict, base_directory: str = None) -> 'PrecipAnalysisData':
        """
        Create an object from a dictionary representation
        :param data: The dictionary
        :param base_directory: The base directory for relative paths
        :return: The object
        """

        ts_data = data['data']

        if isinstance(ts_data, list):
            ts_data = np.array(ts_data)
        elif isinstance(ts_data, dict):
            ts_data = pd.DataFrame(ts_data)

        return cls(
            data=ts_data,
            precipitation_col=data['precipitation_col'] if 'precipitation_col' in data else None,
            inter_event_time=data['inter_event_time'] if 'inter_event_time' in data else 24.0,
            latitude=data['latitude'] if 'latitude' in data else 39.049774180652236,
            longitude=data['longitude'] if 'longitude' in data else -84.66127045790225,
            atlas_14_series=data['atlas_14_series'] if 'atlas_14_series' in data else 'ams',
            sampling_approach=data['sampling_approach'] if 'sampling_approach' in data else cls.SamplingApproach.RANDOM,
            num_event_clusters=data['num_event_clusters'] if 'num_event_clusters' in data else 6
        )


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
            url=f'{SERTODefaults.NOAA_PFDS_REST_SERVER_URL}fe_text_mean.csv',
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
    def cluster_events(
            events: pd.DataFrame,
            cluster_columns: List[str],
            number_of_clusters: int,
            clustering_model: str = 'bayesian_gaussian_mixture',
            *args,
            **kwargs
    ) -> Tuple[pd.DataFrame, Pipeline]:
        """
        This function clusters events based on the specified columns
        :param events: A dataframe with the event attributes
        :param cluster_columns: A list of columns to use for clustering
        :param number_of_clusters: The number of clusters to create
        :param clustering_model: The clustering model to use.
        kmeans, gaussian_mixture, bayesian_gaussian_mixture are supported
        :return: A tuple with a dataframe with the event attributes and cluster labels
        and the clustering model
        """
        from sklearn.cluster import KMeans
        from imblearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler

        model = KMeans(n_clusters=number_of_clusters)

        if clustering_model == 'gaussian_mixture':
            model = GaussianMixture(n_components=number_of_clusters)
        elif clustering_model == 'bayesian_gaussian_mixture':
            model = BayesianGaussianMixture(n_components=number_of_clusters)

        model = Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                (clustering_model, model)
            ]
        )

        cluster_data = events[cluster_columns].copy()
        model.fit(cluster_data)
        cluster_labels = model.predict(cluster_data)
        events['cluster'] = cluster_labels.astype(str)

        if clustering_model == 'bayesian_gaussian_mixture' or clustering_model == 'gaussian_mixture':
            events['probability'] = model.predict_proba(cluster_data).max(axis=1)
            events['likelihood'] = model.score_samples(cluster_data)

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
