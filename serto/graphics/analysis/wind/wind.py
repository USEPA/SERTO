"""

"""
# python imports
from typing import List, Dict, Union, Tuple

import pandas as pd

# third-party imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from imblearn.pipeline import Pipeline
import plotly.figure_factory as ff

# project imports
from ...swmm import SpatialSWMM


class WindVisualization:
    """
    Precipitation visualization class
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_wind_likelihood(
            wind_data: pd.DataFrame,
            model: Pipeline,
            wind_speed_col: str = 'sknt',
            wind_dir_col: str = 'drct',
            wind_direction_bins: int = 128,
            wind_speed_bins: int = 16,
            color_scale: str = 'Jet',
            label_prefix: str = '',
            label_suffix: str = '',
    ) -> go.Barpolar:
        """
        This function plots the wind likelihood
        :param wind_speed_bins:
        :param wind_direction_bins:
        :param wind_data:
        :param model: The wind model
        :param wind_speed_col:
        :param wind_dir_col:
        :param color_scale:
        :param label_prefix:
        :param label_suffix:
        :return:
        """

        traces = []

        min_speed = wind_data[wind_speed_col].min()
        max_speed = wind_data[wind_speed_col].max()
        wind_speed_bins = np.linspace(start=min_speed, stop=max_speed, num=wind_speed_bins)

        if hasattr(px.colors.qualitative, color_scale):
            quad_colors = getattr(px.colors.qualitative, color_scale)
        else:
            quad_colors = getattr(px.colors.sequential, color_scale)

        delta_theta = 360.0 / wind_direction_bins
        theta_bins_labels = np.linspace(start=0.0, stop=360.0, num=wind_direction_bins + 1)
        theta_bins = theta_bins_labels - delta_theta / 2.0
        theta_bins_labels = theta_bins_labels[:-1]

        wind_direction = wind_data[wind_dir_col].values
        wind_direction[wind_direction > theta_bins[-1]] = wind_direction[wind_direction > theta_bins[-1]] - 360.0

        grid = np.meshgrid(wind_speed_bins, theta_bins_labels)
        reshape_grid = np.array(grid).T.reshape(-1, 2)
        plot_field_data = model.score_samples(reshape_grid).T

        # One 1d array all possible speed bins and theta bins for bar plot
        bar_plot = go.Barpolar(
            r=reshape_grid[:, 0],
            theta=reshape_grid[:, 1],
            name=f'{label_prefix}Likelihood{label_suffix}',
            marker=dict(
                color=plot_field_data,
                colorscale=quad_colors,
                colorbar=dict(
                    title='Likelihood',
                    tickformat='#.2g',
                    yanchor='top',
                    y=1,
                    x=0,
                ),
            ),
            showlegend=False,
        )

        return bar_plot

    @staticmethod
    def plot_wind_timeseries(
        wind_data: pd.DataFrame,
        wind_speed_col: str = "sknt",
        wind_dir_col: str = "drct",
        color: str = 'blue',
        resample: str = 'ME',
        arrow_scale: float = 0.25,
        label: str = 'Wind Speed',
        arrowhead_angle = np.pi / 18,
        line_width: int = 3,
        y_label: str = 'Wind Speed (knots)',
        x_label: str = 'Time',
    ) -> go.Figure:
        """
        This function plots the wind timeseries
        :param wind_data: Timeseries of wind data
        :param wind_speed_col: Wind speed column name
        :param wind_dir_col: Wind direction column name
        :param color: The color scale to use
        :param resample: The resample frequency (default is 'ME' for monthly)
        :param arrow_scale: The arrow scale for the wind plot (default is 0.35)
        :param label: The label for the wind plot (default is 'Wind Speed')
        :param arrowhead_angle: The arrowhead angle for the wind plot (default is np.pi / 18)
        :param line_width: The line width for the wind plot (default is 3)
        :param y_label: The y-axis label for the wind plot (default is 'Wind Speed (knots)')
        :param x_label: The x-axis label for the wind plot (default is 'Time')
        :return: The wind timeseries plot
        """

        wind_data_copy = wind_data.copy()

        if resample is not None:
            wind_data_copy = wind_data.resample(resample).mean()

        wind_data_copy['u'] = wind_data_copy[wind_speed_col] * np.cos(np.radians(wind_data_copy[wind_dir_col]))
        wind_data_copy['v'] = wind_data_copy[wind_speed_col] * np.sin(np.radians(wind_data_copy[wind_dir_col]))
        wind_data_copy['r'] = np.sqrt(wind_data_copy['u'] ** 2 + wind_data_copy['v'] ** 2)

        # numeric timestamps in milliseconds
        x = wind_data_copy.index.astype(int) // 10 ** 6
        scale_ratio = (x.max() - x.min()) / (wind_data_copy['r'].max() - wind_data_copy['r'].min())

        wind_fig = ff.create_quiver(
            x=x,
            y=np.zeros(len(wind_data_copy)),
            u=wind_data_copy['u'],
            v=wind_data_copy['v'],
            scale=0.05,
            arrow_scale=arrow_scale,
            angle=arrowhead_angle,
            scaleratio=scale_ratio,
            name=label,
            line=dict(
                width=line_width,
                color=color,
            ),
        )

        empty = go.Scatter(
            x=wind_data_copy.index,
            y=np.zeros(len(wind_data_copy)),
            mode='markers',
            name='Wind Speed',
            showlegend=False,
            marker=dict(
                color=color,
                size=6,
            ),
            visible=True,
        )

        wind_fig = go.Figure(data=[empty, *wind_fig.data])

        wind_fig.update_layout(
            title=label,
            showlegend=True,
            height=600,
            yaxis_title=y_label,
            xaxis_title=x_label,
            xaxis=dict(
                rangeslider=dict(visible=True),
            ),
            legend=dict(
                title=f'Magnitude {y_label}',
            ),
            yaxis=dict(
                title=y_label,
                fixedrange=False,
            ),
        )

        return wind_fig

    @staticmethod
    def plot_wind_rose_bar_polar(
            wind_data: pd.DataFrame,
            wind_speed_col: str = 'sknt',
            wind_dir_col: str = 'drct',
            wind_speed_bins: Union[Tuple[float], List[float]] = None,
            wind_direction_bins: int = 16,
            color_scale: str = 'Plotly',
            label_prefix: str = '',
            label_suffix: str = ' (knots)',
            raxis_mode: str = 'count',
    ) -> List[go.Barpolar]:
        """
        This function plots the wind rose
        :param wind_data: Timeseries of wind data
        :param wind_speed_col: Wind speed column name
        :param wind_dir_col: Wind direction column name
        :param wind_speed_bins: Wind speed bins
        :param wind_direction_bins: Number of wind direction bins
        :param wind_speed_bins: Wind direction bins
        :param color_scale: The color scale to use
        :param label_prefix: The label prefix
        :param label_suffix: The label suffix for the wind speed bins (default is ' (knots)')
        :param raxis_mode: The radial axis mode (default is 'fraction'). Options are 'count' and 'fraction'
        :return: List of wind rose traces
        """

        traces = []

        if wind_speed_bins is None:
            min_speed = wind_data[wind_speed_col].min()
            max_speed = wind_data[wind_speed_col].max()
            wind_speed_bins = np.linspace(start=min_speed, stop=max_speed, num=11)
        else:
            wind_speed_bins = np.array(wind_speed_bins, dtype=np.float32)

        if hasattr(px.colors.qualitative, color_scale):
            quad_colors = getattr(px.colors.qualitative, color_scale)
        else:
            quad_colors = getattr(px.colors.sequential, color_scale)

        delta_theta = 360.0 / wind_direction_bins
        theta_bins_labels = np.linspace(start=0.0, stop=360.0, num=wind_direction_bins + 1)
        theta_bins = theta_bins_labels - delta_theta / 2.0
        theta_bins_labels = theta_bins_labels[:-1]

        wind_direction = wind_data[wind_dir_col].values
        wind_direction[wind_direction > theta_bins[-1]] = wind_direction[wind_direction > theta_bins[-1]] - 360.0

        h, x_edges, y_edges = np.histogram2d(
            x=wind_data[wind_speed_col],
            y=wind_direction,
            bins=[wind_speed_bins, theta_bins],
            density=raxis_mode == 'fraction'
        )

        # normalize the histogram

        # repeat last row to close the circle for h and labels
        # h = np.hstack((h, h[:, [0]]))
        # theta_bins_labels = np.hstack((theta_bins_labels, theta_bins_labels[0]))

        for i in range(len(wind_speed_bins) - 1):
            wind_speed_bin = wind_speed_bins[i]
            wind_speed_bin_next = wind_speed_bins[i + 1]

            traces.append(
                go.Barpolar(
                    r=h[i],
                    theta=theta_bins_labels,
                    name=f'{label_prefix}{wind_speed_bin:#.2g} - {wind_speed_bin_next:#.2g}{label_suffix}',
                    marker=dict(
                        color=quad_colors[i % len(quad_colors)],
                    )
                )
            )

        return traces

    @staticmethod
    def plot_wind_scatter_polar(
            wind_data: pd.DataFrame,
            wind_speed_col: str = 'sknt',
            wind_dir_col: str = 'drct',
            wind_speed_bins: Union[Tuple[float], List[float]] = None,
            wind_direction_bins: int = 16,
            color_scale: str = 'Plotly',
            mode: str = 'lines+markers',
            label_prefix: str = '',
            label_suffix: str = ' (knots)',
            marker_args: Dict = None,
            raxis_mode: str = 'count',
    ) -> List[go.Scatterpolar]:
        """
        This function plots the wind rose
        :param wind_data: Timeseries of wind data
        :param wind_speed_col: Wind speed column name
        :param wind_dir_col: Wind direction column name
        :param wind_speed_bins: Wind speed bins
        :param wind_direction_bins: Number of wind direction bins
        :param color_scale: The color scale to use
        :param mode: The mode for the scatter plot (default is 'markers')
        :param label_prefix: The label prefix
        :param label_suffix: The label suffix for the wind speed bins (default is ' (knots)')
        :param marker_args: The marker arguments for the scatter plot
        :param raxis_mode: The radial axis mode (default is 'count'). Options are 'count' and 'fraction'
        :return: List of wind rose traces
        """

        traces = []

        if marker_args is None:
            marker_args = {}

        if wind_speed_bins is None:
            min_speed = wind_data[wind_speed_col].min()
            max_speed = wind_data[wind_speed_col].max()
            wind_speed_bins = np.linspace(start=min_speed, stop=max_speed, num=11)
        else:
            wind_speed_bins = np.array(wind_speed_bins, dtype=np.float32)

        if hasattr(px.colors.qualitative, color_scale):
            quad_colors = getattr(px.colors.qualitative, color_scale)
        else:
            quad_colors = getattr(px.colors.sequential, color_scale)

        delta_theta = 360.0 / wind_direction_bins
        theta_bins_labels = np.linspace(start=0.0, stop=360.0, num=wind_direction_bins + 1)
        theta_bins = theta_bins_labels - delta_theta / 2.0
        theta_bins_labels = theta_bins_labels[:-1]

        wind_direction = wind_data[wind_dir_col].values
        wind_direction[wind_direction > theta_bins[-1]] = wind_direction[wind_direction > theta_bins[-1]] - 360.0

        h, x_edges, y_edges = np.histogram2d(
            x=wind_data[wind_speed_col],
            y=wind_direction,
            bins=[wind_speed_bins, theta_bins],
            density = raxis_mode == 'fraction'
        )

        # repeat last row to close the circle for h and labels
        h = np.hstack((h, h[:, [0]]))
        theta_bins_labels = np.hstack((theta_bins_labels, theta_bins_labels[0]))

        for i in range(len(wind_speed_bins) - 1):
            wind_speed_bin = wind_speed_bins[i]
            wind_speed_bin_next = wind_speed_bins[i + 1]

            traces.append(
                go.Scatterpolar(
                    r=h[i,:],
                    theta=theta_bins_labels,
                    name=f'{label_prefix}{wind_speed_bin:#.2g} - {wind_speed_bin_next:#.2g}{label_suffix}',
                    mode=mode,
                    marker=dict(
                        color=quad_colors[i % len(quad_colors)],
                        **marker_args
                    )
                )
            )

        return traces
