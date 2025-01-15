"""

"""
# python imports
from typing import List, Dict

import pandas as pd

# third-party imports
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# project imports
from ...swmm import SpatialSWMM


class WindVisualization:
    """
    Precipitation visualization class
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_precipitation(
            precipitation_ts: pd.DataFrame,
            precipitation_events: pd.DataFrame = None,
            plot_clusters: bool = True,
            plot_event_matrix: bool = True,
            even_matrix_fields: List[str] = None,
    ) -> List[go.Figure]:
        """
        This function plots the precipitation event matrix
        :param precipitation_ts: Timeseries of precipitation data
        :param precipitation_events: Timeseries of precipitation events
        :param plot_clusters: Plot precipitation clusters
        :param plot_event_matrix: Plot precipitation event matrix
        :param even_matrix_fields: Fields to plot in the event matrix
        :return:
        """
        #
        # figs = []
        #
        # num_rainfall_columns = precipitation_ts.shape[1]
        #
        # for r in range(num_rainfall_columns):
        #     if precipitation_events is not None:
        #         fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        #         fig.add_trace(go.Scatter(x=precipitation_ts.index, y=precipitation_ts.iloc[:, r], mode='lines', name='Rainfall'), row=1, col=1)
        #         fig.add_trace(go.Scatter
        #     else:
        #         fig = go.Figure()
        #
        #     fig.add_trace(go.Scatter(x=precipitation_ts.index, y=precipitation_ts.iloc[:, r], mode='lines', name='Rainfall'))
        #     fig.update_layout(title='Rainfall', xaxis_title='Time', yaxis_title='Rainfall (mm/hr)')
        #     figs.append(fig)

        return None

