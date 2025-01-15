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
from plotly import figure_factory as ff
import numpy as np

# project imports
from ...swmm import SpatialSWMM


class PrecipitationVisualization:
    """
    Precipitation visualization class
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_precipitation(
            precipitation: pd.DataFrame,
            events: pd.DataFrame = None,
            event_plot_attribute: str = 'precip_total',
            plot_clusters: bool = False,
            y_axis_title: str = 'Event Total Precipitation (in)',
            y2_axis_title: str = 'Precipitation (in)',
            *args, **kwargs
    ) -> List[go.Figure]:
        """
        This function plots the precipitation event matrix
        :param precipitation: Timeseries of precipitation data
        :param events: Timeseries of precipitation events
        :param event_plot_attribute: Event attribute to plot
        :param plot_clusters: Plot precipitation clusters
        :param y_axis_title: Y axis title for the event attribute plotted on the y axis
        :param y2_axis_title: Y axis title for the precipitation attribute plotted on the y2 axis
        :param dvalue: D value for the range breaks in hours
        :return:
        """

        figs = []

        num_rainfall_columns = precipitation.shape[1]

        for r in range(num_rainfall_columns):

            rg_rainfall_name = precipitation.columns[r]

            fig = make_subplots(
                rows=1 if events is None else 2,
                cols=1,
                row_heights=[0.3, 0.7] if events is not None else [1],
                shared_xaxes=True,
                vertical_spacing=0.1
            )

            if events is not None:
                precip_total_column = (num_rainfall_columns, event_plot_attribute) \
                    if (num_rainfall_columns, event_plot_attribute) in events.columns else event_plot_attribute

                events_trace = []
                if plot_clusters:
                    unique_clusters = np.sort(events.cluster.unique())
                    for cluster in unique_clusters:
                        cluster_events = events[events.cluster == cluster]
                        events_trace.append(
                            go.Bar(
                                x=cluster_events.start + cluster_events.duration / 2,
                                y=cluster_events[precip_total_column],
                                base=cluster_events.start,
                                width=cluster_events.duration.dt.total_seconds() * 1000,
                                hovertemplate='<br>'.join([
                                    f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(events.columns)
                                ]),
                                customdata=cluster_events.values,
                                name=f'Cluster {cluster}',
                                showlegend=True,
                            )
                        )
                else:
                    events_trace.append(
                        go.Bar(
                            x=events.start + events.duration / 2,
                            y=events[precip_total_column],
                            base=events.start,
                            width=events.duration.dt.total_seconds() * 1000,
                            hovertemplate='<br>'.join([
                                f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(events.columns)
                            ]),
                            customdata=events.values,
                            name=event_plot_attribute,
                            showlegend=True,
                        )
                    )

                # set y axis title
                fig.update_yaxes(title_text=y_axis_title, row=1, col=1)

                fig.add_traces(events_trace, rows=1, cols=1)

            range_breaks = [
                dict(
                    bounds=[row.actual_end, row.actual_end + row.post_event_time],
                )
                for index, row in events.iterrows()
            ]

            fig.add_trace(
                go.Scatter(
                    x=precipitation.index,
                    y=precipitation[rg_rainfall_name],
                    mode='lines',
                    name=rg_rainfall_name
                ),
                row=1 if events is None else 2,
                col=1
            )
            fig.update_yaxes(title_text=y2_axis_title, row=2, col=1)
            fig.update_layout(
                title=f'Rainfall: {rg_rainfall_name}',
                xaxis2=dict(
                    title='Time',
                    rangebreaks=range_breaks,
                    rangeslider=dict(visible=True),
                ),
            )

            figs.append(fig)

        return figs
