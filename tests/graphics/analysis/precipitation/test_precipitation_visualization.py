# python imports
import unittest

# 3rd party imports
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# project imports
from ....analysis.precipitation import CVG_EXAMPLE_PRECIP_DATA
from serto.analysis.precipitation import PrecipitationAnalysis
from serto.graphics.analysis.precipitation import PrecipitationVisualization
from tests.spatialswmm.swmm import EXAMPLE_SWMM_TEST_MODEL_A


class TestPrecipitationVisualization(unittest.TestCase):

    def setUp(self):
        self.rainfall_data = pd.read_csv(CVG_EXAMPLE_PRECIP_DATA, index_col=0, parse_dates=True)[
                             '2005-01-01':].resample(
            pd.Timedelta(hours=1)).sum()

    def test_rainfall_plotting(self):
        """
        Test the plot_model function
        :return:
        """
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
            cluster_columns=['precip_total', 'precip_peak', 'duration_hours'],
            number_of_clusters=6,
        )

        figs = PrecipitationVisualization.plot_precipitation(
            precipitation=self.rainfall_data,
            events=events,
            plot_clusters=True,
        )

        plotly.offline.plot(figs[0], filename='test_rainfall_plotting.html')

    def test_rainfall_event_plotting(self):
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
