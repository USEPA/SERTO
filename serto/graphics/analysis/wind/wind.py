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
            model: Union[GaussianMixture, BayesianGaussianMixture] = None,
            wind_speed_col: str = 'sknt',
            wind_dir_col: str = 'drct',
            color_scale: str = 'heatmap',
            label_prefix: str = '',
            label_suffix: str = ' (knots)',
            marker_args: Dict = None,
            heatmap_contour_args: Dict = None,
    ) -> List[Union[go.Heatmap, go.Scatterpolar]]:
        """
        This function plots the wind likelihood
        :param wind_data:
        :param model: The wind model
        :param wind_speed_col:
        :param wind_dir_col:
        :param color_scale:
        :param label_prefix:
        :param label_suffix:
        :param marker_args:
        :param heatmap_contour_args:
        :return:
        """

        if marker_args is None:
            marker_args = {}

        if heatmap_contour_args is None:
            heatmap_contour_args = dict()

        traces: List[Union[go.Heatmap, go.Scatterpolar]] = [go.Scatterpolar(
            r=wind_data[wind_speed_col],
            theta=wind_data[wind_dir_col],
            name=f'{label_prefix}Wind Speed and Direction{label_suffix}',
            marker=dict(
                **marker_args
            )
        ), go.Heatmap(
            z=wind_data[wind_speed_col],
            x=wind_data[wind_dir_col],
            y=wind_data[wind_speed_col],
            colorscale=color_scale,
            name=f'{label_prefix}Wind Speed and Direction{label_suffix}',
            **heatmap_contour_args
        )]

        return traces

    @staticmethod
    def plot_wind_rose_bar_polar(
            wind_data: pd.DataFrame,
            wind_speed_col: str = 'sknt',
            wind_dir_col: str = 'drct',
            wind_speed_bins: Union[Tuple[float], List[float]] = None,
            num_wind_dir_bins: int = 16,
            color_scale: str = 'Plotly',
            label_prefix: str = '',
            label_suffix: str = ' (knots)',
    ) -> List[go.Barpolar]:
        """
        This function plots the wind rose
        :param wind_data: Timeseries of wind data
        :param wind_speed_col: Wind speed column name
        :param wind_dir_col: Wind direction column name
        :param wind_speed_bins: Wind speed bins
        :param num_wind_dir_bins: Number of wind direction bins
        :param wind_speed_bins: Wind direction bins
        :param color_scale: The color scale to use
        :param label_prefix: The label prefix
        :param label_suffix: The label suffix for the wind speed bins (default is ' (knots)')
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

        delta_theta = 360.0 / num_wind_dir_bins
        theta_bins_labels = np.linspace(start=0.0, stop=360.0, num=num_wind_dir_bins + 1)
        theta_bins = theta_bins_labels - delta_theta / 2.0
        theta_bins_labels = theta_bins_labels[:-1]

        wind_direction = wind_data[wind_dir_col].values
        wind_direction[wind_direction > theta_bins[-1]] = wind_direction[wind_direction > theta_bins[-1]] - 360.0

        h, x_edges, y_edges = np.histogram2d(
            x=wind_data[wind_speed_col],
            y=wind_direction,
            bins=[wind_speed_bins, theta_bins]
        )

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
            num_wind_dir_bins: int = 16,
            color_scale: str = 'Plotly',
            mode: str = 'lines+markers',
            label_prefix: str = '',
            label_suffix: str = ' (knots)',
            marker_args: Dict = None,
    ) -> List[go.Scatterpolar]:
        """
        This function plots the wind rose
        :param wind_data: Timeseries of wind data
        :param wind_speed_col: Wind speed column name
        :param wind_dir_col: Wind direction column name
        :param wind_speed_bins: Wind speed bins
        :param num_wind_dir_bins: Number of wind direction bins
        :param color_scale: The color scale to use
        :param mode: The mode for the scatter plot (default is 'markers')
        :param label_prefix: The label prefix
        :param label_suffix: The label suffix for the wind speed bins (default is ' (knots)')
        :param marker_args: The marker arguments for the scatter plot
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

        delta_theta = 360.0 / num_wind_dir_bins
        theta_bins_labels = np.linspace(start=0.0, stop=360.0, num=num_wind_dir_bins + 1)
        theta_bins = theta_bins_labels - delta_theta / 2.0
        theta_bins_labels = theta_bins_labels[:-1]

        wind_direction = wind_data[wind_dir_col].values
        wind_direction[wind_direction > theta_bins[-1]] = wind_direction[wind_direction > theta_bins[-1]] - 360.0

        h, x_edges, y_edges = np.histogram2d(
            x=wind_data[wind_speed_col],
            y=wind_direction,
            bins=[wind_speed_bins, theta_bins]
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
