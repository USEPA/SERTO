"""
serto.analysis.wind.WindAnalysis contains the functionality for wind analysis.
"""
# python imports
from typing import List, Dict, Union, Tuple, Optional

# third-party imports
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from numpy.typing import NDArray

# local imports
from ... import IDictable


class WindAnalysisData(IDictable):
    """
    This class contains the wind data to be used for wind analysis
    """

    def __init__(
            self,
            data: Union[str, NDArray, pd.DataFrame] = None,
            wind_speed_col: str = None,
            wind_direction_col: str = None,
    ):
        """
        Constructor for the wind analysis data object.
        :param data: The wind data to use for the analysis (either a path to a CSV file or a numpy array of dimensions
        (n_samples, 2) where the first column is the wind speed and the second column is the wind direction)
        :param wind_speed_col: The wind speed column name if the data is a pandas DataFrame
        :param wind_direction_col: The wind direction column name if the data is a pandas DataFrame
        """
        self._data = data
        self._wind_speed_col = wind_speed_col
        self.wind_direction_col = wind_direction_col

        if isinstance(data, pd.DataFrame):
            self._model_data = data[[wind_speed_col, wind_direction_col]]
        elif isinstance(data, np.ndarray):
            self._model_data = data
        elif isinstance(data, str):
            self._model_data = pd.read_csv(data, index_col=0, parse_dates=True)[[wind_speed_col, wind_direction_col]]
        else:
            raise ValueError('Invalid data type')

    @property
    def data(self) -> Union[NDArray, pd.DataFrame]:
        """
        Get the wind data for the analysis object
        :return:
        """
        return self._model_data

    def to_dict(self, base_directory: str = None) -> dict:
        """
        Convert the object to a dictionary
        :param base_directory: The base directory for relative paths
        :return: The dictionary
        """

        if isinstance(self._data, pd.DataFrame):
            data = self._model_data.to_dict()
        elif isinstance(self._data, np.ndarray):
            data = self._model_data.to_list()
        elif isinstance(self._data, str):
            data = self._data

        data_dict = {
            'data': data,
            'wind_speed_col': self._wind_speed_col,
            'wind_direction_col': self.wind_direction_col,
        }

        return data_dict

    @classmethod
    def from_dict(cls, data: dict, base_directory: str = None) -> 'WindAnalysisData':
        """
        Create an object from a dictionary representation
        :param data: The dictionary
        :param base_directory: The base directory for relative paths
        :return: The object
        """

        tdata = data['data']

        if isinstance(tdata, dict):
            arg_data = pd.DataFrame(tdata)
        elif isinstance(tdata, list):
            arg_data = np.array(tdata)
        else:
            arg_data = tdata

        return cls(
            data=arg_data,
            wind_speed_col=data['wind_speed_col'] if 'wind_speed_col' in data else None,
            wind_direction_col=data['wind_direction_col'] if 'wind_direction_col' in data else None,
        )


class WindAnalysis(IDictable):
    """
    This class performs wind analysis
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
            wind_data: WindAnalysisData,
            sampling_approach: Optional[SamplingApproach] = SamplingApproach.RANDOM,
            *args,
            **kwargs,
    ) -> None:
        """
        Constructor for the wind analysis object
        :param sampling_approach: The sampling approach to use for the wind analysis
        :param wind_speed: Optional wind speed data
        :param wind_direction: Optional wind direction data
        :param args: Additional arguments for the wind analysis
        :param kwargs: Additional keyword arguments for the wind analysis
        """
        self.sampling_approach = sampling_approach
        self.wind_data = wind_data
        self.model_args = args
        self.model_kwargs = kwargs
        self.model = None
        self._fit()

    def _fit(self) -> None:
        """
        This function fits the wind data
        :param x: The wind data
        :return: None
        """

        if self.sampling_approach == self.SamplingApproach.GAUSSIAN_MIXTURE:
            self.model = GaussianMixture(*self.model_args, **self.model_kwargs)
        elif self.sampling_approach == self.SamplingApproach.BAYESIAN_GAUSSIAN_MIXTURE:
            self.model = BayesianGaussianMixture(*self.model_args, **self.model_kwargs)
            self.model.fit(self.wind_data.data)

    def sample(self, num_samples: int) -> NDArray:
        """
        This function samples the wind data
        :param num_samples: The number of samples to take
        :return: The sampled wind data
        """
        if self.sampling_approach == self.SamplingApproach.RANDOM:
            sample_idx = np.random.choice(self.wind_data.data.shape[0], num_samples, *self.model_args, **self.model_kwargs)
            samples = self.wind_data.data[sample_idx]
        else:
            samples = self.model.sample(num_samples)

        return samples

    def to_dict(self, base_directory: str = None) -> dict:
        """
        Convert the object to a dictionary
        :param base_directory: The base directory for relative paths
        :return: The dictionary
        """

        arguments = {
            'wind_data': self.wind_data.to_dict(base_directory),
            'sampling_approach': self.sampling_approach,
            'args': self.model_args,
            'kwargs': self.model_kwargs,
        }

        return arguments

    @classmethod
    def from_dict(cls, data: dict, base_directory: str = None) -> 'WindAnalysis':
        """
        Create an object from a dictionary representation
        :param data: The dictionary
        :param base_directory: The base directory for relative paths
        :return: The object
        """

        wind_data = WindAnalysisData.from_dict(data['wind_data'], base_directory)

        return cls(
            wind_data=wind_data,
            sampling_approach=data['sampling_approach'],
            *data['args'],
            **data['kwargs'],
        )

    @staticmethod
    def joint_speed_direction_gmm_model(
            wind_data: pd.DataFrame,
            wind_speed_col: str = 'sknt',
            wind_dir_col: str = 'drct',
            gmm_type: str = 'BayesianGaussianMixture',
            weight_concentration_prior_type: str = 'dirichlet_distribution',
            n_components: int = 6,
            covariance_type: str = 'full',
    ) -> Pipeline:
        """
        This function creates a joint speed and direction distribution model
        :param wind_data: The wind data to model
        :param wind_speed_col: The wind speed column name
        :param wind_dir_col: The wind direction column name
        :param gmm_type: The type of GMM to use. Available models are
        GaussianMixture and BayesianGaussianMixture
        :param weight_concentration_prior_type: The weight concentration prior
        type for the BayesianGaussianMixture model
        :param n_components: The number of components for the model
        :param covariance_type: The covariance type for the model
        :return: None
        """

        if gmm_type == 'BayesianGaussianMixture':
            model = Pipeline(
                [
                    ('scaler', MinMaxScaler()),
                    ('model', BayesianGaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        weight_concentration_prior_type=weight_concentration_prior_type,
                    ))
                ]
            )
        else:
            model = Pipeline(
                [
                    ('scaler', MinMaxScaler()),
                    ('model', GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                    ))
                ]
            )

        model.fit(wind_data[[wind_speed_col, wind_dir_col]])

        return model
