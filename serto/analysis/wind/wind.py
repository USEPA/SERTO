"""
serto.analysis.wind.WindAnalysis contains the functionality for wind analysis.
"""
# python imports
import pandas as pd


# third-party imports
from sklearn.mixture import GaussianMixture

# local imports


class WindAnalysis:
    """
    This class performs wind analysis
    """

    @staticmethod
    def joint_speed_direction_distribution_model(
            wind_data: pd.DataFrame,
            direction: str,
            speed: str,
            n_components: int = 6,
            covariance_type: str = 'full',
    ) -> GaussianMixture:
        """
        This function creates a joint speed and direction distribution model
        :param wind_data: The wind data to model
        :param direction: The wind direction column name
        :param speed: The wind speed column name
        :param n_components: The number of components for the model
        :param covariance_type: The covariance type for the model
        :return: None
        """
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
        )

        model.fit(wind_data[[direction, speed]])

        return model

    @staticmethod
    def probabilities(
            wind_data: pd.DataFrame,
            direction: str,
            speed: str,
            output_file: str,
    ):
        """
        This function creates a wind rose plot
        :param wind_data: The wind data to plot
        :param direction: The wind direction column name
        :param speed: The wind speed column name
        :param output_file: The output file for the wind rose plot
        :return: None
        """
        pass

    @staticmethod
    def sample(
            wind_data: pd.DataFrame,
            direction: str,
            speed: str,
            num_samples: int,
    ):
        """
        This function samples the wind data
        :param wind_data: The wind data to sample from
        :param direction: The wind direction column name
        :param speed: The wind speed column name
        :param num_samples: The number of samples to take from the wind data
        :return: The sampled wind data as a pandas DataFrame
        """
        pass


