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

    def test_polar(self):
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

    def test_plot_probabilities(self):
        """
        Test the plot_model function
        :return:
        """

        cluster_columns = ['precip_total', 'precip_peak', 'duration_hours']

        events = PrecipitationAnalysis.get_events(
            rainfall=self.rainfall_data,
            inter_event_time=pd.Timedelta(days=1),
            floor=1.0E-13
        )

        events = PrecipitationAnalysis.get_noaa_event_return_periods(
            events=events,
            latitude=39.049774180652236,
            longitude=-84.66127045790225,
            series='ams'
        )

        events['duration_hours'] = events['duration'] / pd.Timedelta(hours=1)

        events, cluster_model = PrecipitationAnalysis.cluster_events(
            events=events,
            cluster_columns=cluster_columns,
            number_of_clusters=6,
        )

        figs = PrecipitationVisualization.plot_events(
            events=events,
            plot_clusters=True,
            event_plot_attributes=cluster_columns,
            event_plot_attribute_labels=['Total Precipitation (in)', 'Peak Intensity (in/hr)', 'Duration (hours)']
        )

        plotly.offline.plot(figs[0][0], filename='test_rainfall_event_plotting.html')
        plotly.offline.plot(figs[0][1], filename='test_rainfall_event_distribution_plotting.html')
