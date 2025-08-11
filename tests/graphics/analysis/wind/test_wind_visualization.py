# python imports
import unittest

# 3rd party imports
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# project imports
from ....analysis.wind import EXAMPLE_CVG_WIND_DATA
from serto.analysis.wind import WindAnalysis
from serto.graphics.analysis.wind import WindVisualization


class TestWindVisualization(unittest.TestCase):

    def setUp(self):
        self.wind_data = pd.read_csv(
            EXAMPLE_CVG_WIND_DATA,
            index_col="TimeEST",
            parse_dates=["TimeEST"]
        )

        self.wind_data = self.wind_data.drop(columns=['station', 'station_name'])

        # remove duplicate indexes and sort
        self.wind_data = self.wind_data[~self.wind_data.index.duplicated(keep='first')]
        self.wind_data = self.wind_data.sort_index()

    def test_wind_polar_plots(self):
        """
        Test the plot_model function
        :return:
        """
        wind_rose_scatter = WindVisualization.plot_wind_scatter_polar(
            wind_data=self.wind_data,
            wind_speed_col='sknt',
            wind_dir_col='drct',

        )

        wind_rose_polar = WindVisualization.plot_wind_rose_bar_polar(
            wind_data=self.wind_data,
            wind_speed_col='sknt',
            wind_dir_col='drct',
        )

        fig = go.Figure([*wind_rose_polar, *wind_rose_scatter])

        fig.update_layout(
            legend=dict(
                title='Wind Speed (knots)',
            )
        )
        plotly.offline.plot(fig, filename='test_wind_rose_scatter_polar.html')

    def test_wind_statistics_plot(self):
        """
        Test the plot_model function
        :return:
        """

        wind_clustering_model = WindAnalysis.joint_speed_direction_gmm_model(
            wind_data=self.wind_data,
            n_components=12
        )

        probability_plots = WindVisualization.plot_wind_likelihood(
            wind_data=self.wind_data,
            model=wind_clustering_model,
        )

        fig = go.Figure(probability_plots)
        plotly.offline.plot(fig, filename='test_wind_statistics_plot.html')