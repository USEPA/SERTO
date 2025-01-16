"""

"""
# python imports
from typing import List, Dict, Tuple, Union

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
    def __convert_splom_to_scatter(splom: go.Splom) -> go.Scatter:
        scatter = go.Scatter(
            x=splom['dimensions'][0]['values'],
            y=splom['dimensions'][1]['values'],
            mode='markers',
            marker=splom['marker'].to_plotly_json(),
            name=splom['name'],
            opacity=splom['opacity'],
            showlegend=splom['showlegend'],
            legendgroup=splom['legendgroup'],
            legendrank=splom['legendrank'],
            legendwidth=splom['legendwidth'],
            customdata=splom['customdata'],
            hoverinfo=splom['hoverinfo'],
            hovertemplate=splom['hovertemplate'],
            hovertext=splom['hovertext'],
            ids=splom['ids'],
            selected=splom['selected'].to_plotly_json(),
            selectedpoints=splom['selectedpoints'],
            unselected=splom['unselected'].to_plotly_json(),
            visible=splom['visible'],
            xhoverformat=splom['xhoverformat'],
            yhoverformat=splom['yhoverformat'],
        )
        return scatter

    @staticmethod
    def plot_events(
            events: pd.DataFrame,
            rain_gauge_names: List[str] = None,
            event_plot_attributes: List[str] = None,
            event_plot_attribute_labels: List[str] = None,
            plot_clusters: bool = False,
            *args, **kwargs
    ) -> List[Tuple[go.Figure, Union[go.Figure, None]]]:
        """
        This function plots the precipitation events matrix for given rain gauge names
        :param events: DataFrame of events data
        :param rain_gauge_names:  List of rain gauge names
        :param event_plot_attributes:  List of event plot attributes
        :param event_plot_attribute_labels:  List of event plot attribute labels
        :param plot_clusters:
        :return:
        """

        local_events = events.copy()

        figures: List[Tuple[go.Figure, Union[go.Figure, None]]] = []

        if rain_gauge_names is not None:
            for rain_gauge_name in rain_gauge_names:
                event_plot_attributes = [
                    (rain_gauge_name, event_plot_attribute) for event_plot_attribute in event_plot_attributes
                    if (rain_gauge_name, event_plot_attribute) in events.columns
                ]

                # Error if no event plot attributes is empty
                if len(event_plot_attributes) == 0:
                    raise ValueError('No event plot attributes found in the events DataFrame')
                else:
                    if 'cluster' in local_events.columns and plot_clusters:
                        plot_events = local_events[[*event_plot_attributes, 'cluster']]
                    else:
                        plot_events = local_events[event_plot_attributes]

                    if event_plot_attribute_labels is not None:
                        if len(event_plot_attributes) == len(event_plot_attribute_labels):
                            plot_events.rename(
                                columns=dict(zip(event_plot_attributes, event_plot_attribute_labels)),
                                inplace=True
                            )
                            event_plot_attributes = event_plot_attribute_labels
                        else:
                            raise ValueError('Event plot attributes and labels must be the same length')

                    category_orders = dict(cluster=np.sort(
                        plot_events.cluster.unique())) if 'cluster' in plot_events.columns and plot_clusters else None

                    fig = px.scatter_matrix(
                        plot_events,
                        dimensions=event_plot_attributes,
                        color='cluster' if 'cluster' in plot_events.columns and plot_clusters else None,
                        category_orders=category_orders,
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        title=f'Rainfall Events for Gauge: {rain_gauge_name}'
                    )

                    summary_fig = []

                    if 'cluster' in local_events.columns and plot_clusters:
                        plot_events = plot_events.sort_values('cluster')
                        for f, event_plot_attribute in enumerate(event_plot_attributes):
                            summary_fig.append(
                                go.Box(
                                    x=plot_events['cluster'],
                                    y=plot_events[event_plot_attribute],
                                    name=f'{event_plot_attribute}',
                                    hovertemplate='<br>'.join([
                                        f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in
                                        enumerate(plot_events.columns)
                                    ]),
                                    customdata=plot_events.values,
                                    boxpoints='all',
                                    notched=True,
                                    showwhiskers=True,
                                )
                            )

                        summary_fig = go.Figure(summary_fig)

                        layout = dict(
                            yaxis=dict(
                                range=[0, None],
                                showline=True,
                            )
                        )

                        summary_fig.update_layout(
                            xaxis=dict(
                                title='Clusters',
                            ),
                            barmode='group',
                            **layout
                        )

                    figures.append((fig, summary_fig))

        else:

            if 'cluster' in local_events.columns and plot_clusters:
                plot_events = local_events[[*event_plot_attributes, 'cluster']]
            else:
                plot_events = local_events[event_plot_attributes]

            if event_plot_attribute_labels is not None:
                if len(event_plot_attributes) == len(event_plot_attribute_labels):
                    plot_events.rename(
                        columns=dict(zip(event_plot_attributes, event_plot_attribute_labels)),
                        inplace=True
                    )
                    event_plot_attributes = event_plot_attribute_labels
                else:
                    raise ValueError('Event plot attributes and labels must be the same length')

            category_orders = dict(cluster=np.sort(
                plot_events.cluster.unique())) if 'cluster' in plot_events.columns and plot_clusters else None

            fig = px.scatter_matrix(
                plot_events,
                category_orders=category_orders,
                dimensions=event_plot_attributes,
                color='cluster' if 'cluster' in plot_events.columns and plot_clusters else None,
                title=f'Rainfall Events',
            )

            fig.update_traces(diagonal_visible=False)

            summary_fig = []

            if 'cluster' in local_events.columns and plot_clusters:
                plot_events = plot_events.sort_values('cluster')
                for f, event_plot_attribute in enumerate(event_plot_attributes):
                    summary_fig.append(
                        go.Box(
                            x=plot_events['cluster'],
                            y=plot_events[event_plot_attribute],
                            name=f'{event_plot_attribute}',
                            hovertemplate='<br>'.join([
                                f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(plot_events.columns)
                            ]),
                            customdata=plot_events.values,
                            showlegend=True,
                            boxpoints='all',
                            notched=True,
                            showwhiskers=True,
                        )
                    )

                summary_fig = go.Figure(summary_fig)

                layout = dict(
                    yaxis=dict(
                        range=[0, None],
                        showline=True,
                    )
                )

                summary_fig.update_layout(
                    xaxis=dict(
                        title='Clusters',
                    ),
                    barmode='group',
                    **layout
                )

            figures.append((fig, summary_fig))

        return figures

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
