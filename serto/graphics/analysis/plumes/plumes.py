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
import contextily as ctx
import matplotlib.pyplot as plt

# project imports
from ...swmm import SpatialSWMM, SpatialSWMMVisualization
from ....analysis.wind import WindAnalysis

from ....analysis.plumes import GaussianPlume, PlumeEventMatrix
from ....graphics.analysis.wind import WindVisualization

class GaussianPlumeVisualization:
    """
    Gaussian plume visualization class
    """

    def __init__(self):
        pass


    @staticmethod
    def plot_plume_event(
            swmm_model: SpatialSWMM,
            wind_analysis: WindAnalysis,
            plume: GaussianPlume,
    ) -> go.Figure:
        pass
class PlumeEventMatrixVisualization:
    """
    Plume event matrix visualization class
    """
    def __init__(self):
        pass

    @staticmethod
    def plot_plume_event_matrix(plume_event_matrix: PlumeEventMatrix, event_indexes: List[int] = None) -> List[go.Figure]:
        """
        This function plots the plume event matrix
        :param plume_event_matrix:
        :return:
        """

        if event_indexes is None:
            event_indexes = list(range(len(plume_event_matrix.plume_events)))


        figs = []

        catchment_traces = SpatialSWMMVisualization.plot_catchments_plotly(plume_event_matrix.swmm_model)
        node_traces = SpatialSWMMVisualization.plot_nodes_plotly(plume_event_matrix.swmm_model)

        nodes_centroid = plume_event_matrix.swmm_model.nodes.dissolve().centroid.to_crs('EPSG:4326')
        lon = nodes_centroid.geometry.x[0]
        lat = nodes_centroid.geometry.y[0]

        for event_index in event_indexes:

            fig = go.Figure()
            fig.add_traces(catchment_traces)
            fig.add_traces(node_traces)
            row = plume_event_matrix.plume_events_summary.loc[event_index]
            name = 'Attributes: ' + ', '.join([f'{k}: {v}' for k, v in row.items()])


            plume_event = plume_event_matrix.get_plume_event(event_index)

            plume_event_concentrations = plume_event.plume_mesh(
                x_min=plume_event_matrix.model_bounds['min_x'],
                x_max=plume_event_matrix.model_bounds['max_x'],
                y_min=plume_event_matrix.model_bounds['min_y'],
                y_max=plume_event_matrix.model_bounds['max_y'],
                resolution=plume_event_matrix.contaminant_loading_spatial_resolution,
                crs=plume_event_matrix.swmm_model._crs
            )

            plume_event_concentrations = plume_event_concentrations.to_crs('EPSG:4326')
            valid_concentrations = plume_event_concentrations[plume_event_concentrations['concentration'] > 1.0e-12]
            valid_concentrations_to_geo_crs_data = pd.DataFrame(valid_concentrations.drop(columns='geometry'))

            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=valid_concentrations.__geo_interface__,
                    locations=valid_concentrations.index,
                    z=valid_concentrations['concentration'],
                    colorscale=[[0, 'blue'], [1, 'red']],
                    hovertemplate='<br>'.join([
                        f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(valid_concentrations_to_geo_crs_data.columns)
                    ]),
                    text=name,
                    customdata=valid_concentrations_to_geo_crs_data.values,
                    showscale=True,
                    # Scale title
                    colorbar=dict(
                        title=f'Concentration [{plume_event_matrix.contaminant_units}]',
                    ),
                    showlegend=True,
                    marker=dict(
                        opacity=0.3,
                    )
                )
            )

            fig.update_layout(
                title=f'Plume Event {event_index}',
                map_style="carto-positron",
                showlegend=True,
                map={
                    'center': {'lon': lon, 'lat': lat},
                    'zoom': 13
                }
            )

            figs.append(fig)

        return figs