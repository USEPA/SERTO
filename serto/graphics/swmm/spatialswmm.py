# python imports
import os
from typing import List, Dict, Any, TypeVar, Tuple

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
from ... import IDictable
from ...swmm import SpatialSWMM

T = TypeVar("T", bound="MyClass")


class SpatialSWMMVisualization(IDictable):
    def __init__(self, inp: str, crs: str, output: str, *args, **kwargs):
        """
        Initialize the spatial swmm visualization object
        :param inp: SWMM input file
        :param crs: coordinate reference system
        :param output: Visualization output file path
        :param args:
        :param kwargs:
        """
        self.inp = inp
        self.crs = crs
        self.output = output

    def plot(self):
        """
        Generate the spatial swmm model
        :return:
        """
        swmm_spatial_model = SpatialSWMM.read_model(model_path=self.inp, crs=self.crs)

        # Check if output file extension is html for plotly otherwise use matplotlib
        if self.output is not None and self.output.endswith('.html'):

            fig = go.Figure()

            catchment_traces = SpatialSWMMVisualization.plot_catchments_plotly(swmm_spatial_model)
            link_traces = SpatialSWMMVisualization.plot_links_plotly(swmm_spatial_model)
            node_traces = SpatialSWMMVisualization.plot_nodes_plotly(swmm_spatial_model)

            for catchment_trace in catchment_traces:
                fig.add_trace(catchment_trace)

            for node_trace in node_traces:
                fig.add_trace(node_trace)

            nodes_centroid = swmm_spatial_model.nodes.dissolve().centroid.to_crs('EPSG:4326')
            lon = nodes_centroid.geometry.x[0]
            lat = nodes_centroid.geometry.y[0]

            fig.update_layout(
                title=f'',
                map_style="carto-positron",
                showlegend=True,
                # width=1600,
                # height=800,
                map={
                    'center': {'lon': lon, 'lat': lat},
                    'zoom': 10
                }
            )

            plotly.offline.plot(fig, filename=self.output)

        else:
            plt.rcParams['figure.figsize'] = [8, 12]
            plt.rcParams["figure.dpi"] = 300
            ax = swmm_spatial_model.subcatchments.plot(
                linewidth=0.15,
                edgecolor='black',
                facecolor='salmon',
                alpha=0.5,
                zorder=1
            )

            link_styles = {
                'CONDUITS': 'solid',
                'PUMPS': 'dashed',
                'ORIFICES': 'dashdot',
                'WEIRS': 'dotted',
            }

            link_markers = {
                'CONDUITS': None,
                'PUMPS': '1',
                'ORIFICES': 'o',
                'WEIRS': '1',
            }

            line_colors = {
                'CONDUITS': 'blue',
                'PUMPS': 'red',
                'ORIFICES': 'purple',
                'WEIRS': 'black',
            }

            line_widths = {
                'CONDUITS': 0.45,
                'PUMPS': 4,
                'ORIFICES': 4,
                'WEIRS': 4,
            }

            for link_type, linestyle in link_styles.items():
                ax = swmm_spatial_model.links[swmm_spatial_model.links['LinkType'] == link_type].plot(
                    ax=ax,
                    linewidth=line_widths[link_type],
                    color=line_colors[link_type],
                    markersize=10,
                    label=link_type,
                    zorder=2,
                    capstyle='butt',
                )

            node_markers = {
                'JUNCTIONS': '.',
                'OUTFALLS': 'v',
                'STORAGES': 's',
                'DIVIDERS': 'o',
            }

            for node_type, marker in node_markers.items():
                ax = swmm_spatial_model.nodes[swmm_spatial_model.nodes['NodeType'] == node_type].plot(
                    ax=ax,
                    marker=marker,
                    markersize=20,
                    label=node_type,
                    zorder=3,
                )

            plt.legend(title='Elements', loc='upper right')
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=swmm_spatial_model._crs)
            plt.savefig(self.output)

    def to_dict(self, base_directory: str = None) -> dict:
        """
        Export the spatial swmm visualization object to a dictionary
        :return:
        """
        inp = self.inp
        output = self.output
        
        if base_directory is not None:
            if os.path.isabs(inp):
                inp = os.path.relpath(inp, base_directory)
            
            if os.path.isabs(output):
                output = os.path.relpath(output, base_directory)

            

        return {
            'inp': input,
            'crs': self.crs,
            'output': output
        }

    @classmethod
    def from_dict(cls, data: Dict[Any, Any], base_directory: str = None) -> 'SpatialSWMMVisualization':
        """
        Import the spatial swmm visualization object from a dictionary
        :param data: Dictionary containing the spatial swmm visualization object
        :return: SpatialSWMMViz object
        """
        if base_directory is not None:
            if 'inp' in data and not os.path.isabs(data['inp']):
                data['inp'] = os.path.join(base_directory, data['inp'])
            if 'output' in data and not os.path.isabs(data['output']):
                data['output'] = os.path.join(base_directory, data['output'])

        spatial_swmm_viz = SpatialSWMMVisualization(**data)
        return  spatial_swmm_viz

    @staticmethod
    def plot_nodes_plotly(
            swmm_model: SpatialSWMM,
            marker_symbols: Dict[str, str] = None,
            hover_display_fields: List[str] = None,
            *args, **kwargs
    ) -> List[go.Scattermap]:
        """
        This function plots the nodes of the SWMM model
        :param swmm_model: SpatialSWMM object
        :param marker_symbols: Dictionary containing the marker symbols for the nodes

        For example:

        ```
            {
                "JUNCTIONS": {
                "symbol": "circle",
                "color": "blue",
                "size": 5
                },
            }
        ```
        :param hover_display_fields: List of fields to display in the hover tooltip
        :return:
        """
        scatter_maps: List[go.Scattermap] = []

        nodes_to_geo_crs = swmm_model.nodes.to_crs('EPSG:4326')
        nodes_to_geo_crs.insert(0, 'Name', nodes_to_geo_crs.index)

        # extract lat lon from geometry and append as columns
        nodes_to_geo_crs['lon'] = nodes_to_geo_crs.geometry.x
        nodes_to_geo_crs['lat'] = nodes_to_geo_crs.geometry.y

        if marker_symbols is None:
            marker_symbols = {
                "JUNCTIONS": {
                    "symbol": "circle",
                    "color": "blue",
                    "size": 5
                },
                "OUTFALLS": {
                    "symbol": "circle",
                    "color": "red",
                    "size": 8
                },
                "DIVIDER": {
                    "symbol": "circle",
                    "color": "purple",
                    "size": 8
                },
                "STORAGE": {
                    "symbol": "circle",
                    "color": "green",
                    "size": 10
                }
            }
        else:
            if "JUNCTION" not in marker_symbols:
                marker_symbols["JUNCTIONS"] = {
                    "symbol": "circle",
                    "color": "blue",
                    "size": 5
                }
            if "OUTFALLS" not in marker_symbols:
                marker_symbols["OUTFALLS"] = {
                    "symbol": "circle",
                    "color": "red",
                    "size": 8
                }
            if "DIVIDER" not in marker_symbols:
                marker_symbols["DIVIDER"] = {
                    "symbol": "circle",
                    "color": "purple",
                    "size": 8
                }

            if "STORAGE" not in marker_symbols:
                marker_symbols["STORAGE"] = {
                    "symbol": "circle",
                    "color": "green",
                    "size": 10
                }

        nodes_to_geo_crs = pd.DataFrame(nodes_to_geo_crs.drop(columns='geometry'))
        junctions = nodes_to_geo_crs[nodes_to_geo_crs['NodeType'] == 'JUNCTIONS']
        outfalls = nodes_to_geo_crs[nodes_to_geo_crs['NodeType'] == 'OUTFALLS']
        dividers = nodes_to_geo_crs[nodes_to_geo_crs['NodeType'] == 'DIVIDER']
        storage = nodes_to_geo_crs[nodes_to_geo_crs['NodeType'] == 'STORAGE']

        # if junctions has elements create geo.Scattermap with color blue and shape circle with hover data as table of all
        if not junctions.empty:
            scatter_maps.append(
                go.Scattermap(
                    name='Junctions',
                    lat=junctions['lat'],
                    lon=junctions['lon'],
                    mode='markers',
                    marker=marker_symbols['JUNCTIONS'],
                    hovertemplate='<br>'.join([
                        f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(junctions.columns[:10])
                    ]),

                    text=junctions['Name'],
                    customdata=junctions.values[:, :10],
                )
            )

        # Triangle red for outfalls
        if not outfalls.empty:
            scatter_maps.append(
                go.Scattermap(
                    name='Outfalls',
                    lat=outfalls['lat'],
                    lon=outfalls['lon'],
                    mode='markers',
                    marker=marker_symbols['OUTFALLS'],
                    hovertemplate='<br>'.join([
                        f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(outfalls.columns[:5])
                    ]),
                    text=outfalls['Name'],
                    customdata=outfalls.values[:, 5],
                )
            )

        # Diamond purple for dividers
        if not dividers.empty:
            scatter_maps.append(
                go.Scattermap(
                    name='Dividers',
                    lat=dividers['lat'],
                    lon=dividers['lon'],
                    mode='markers',
                    marker=marker_symbols['DIVIDER'],
                    hovertemplate='<br>'.join([
                        f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(dividers.columns[:10])
                    ]),
                    text=dividers['Name'],
                    customdata=dividers.values[:,:10],
                )
            )

        # Square light blue for storage
        if not storage.empty:
            scatter_maps.append(
                go.Scattermap(
                    name='Storage',
                    lat=storage['lat'],
                    lon=storage['lon'],
                    mode='markers',
                    marker=marker_symbols['STORAGE'],
                    hovertemplate='<br>'.join([
                        f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(storage.columns[:10])
                    ]),
                    text=storage['Name'],
                    customdata=storage.values[:,:10],
                )
            )

        return scatter_maps

    @staticmethod
    def plot_catchments_plotly(
            swmm_model: SpatialSWMM,
            display_fields: List[str] = None,
            color_scale: List[Tuple[float, str]] = None,
            marker_style: Dict[str, Any] = None,
            *args,
            **kwargs
    ) -> List[go.Choroplethmap]:
        """
        This function plots the catchments of the SWMM model
        :param swmm_model:
        :param args:
        :param kwargs:
        :return:
        """
        choropleth_maps: List[go.Choroplethmap] = []

        subcatchments_to_geo_crs = swmm_model.subcatchments.to_crs('EPSG:4326')
        subcatchments_to_geo_crs.insert(0, 'Name', subcatchments_to_geo_crs.index)

        # if subcatchments has elements create geo.Choroplethmap with color blue and hover data as table of all
        subcatchments_to_geo_crs_data = pd.DataFrame(subcatchments_to_geo_crs.drop(columns='geometry'))

        if not subcatchments_to_geo_crs.empty:
            choropleth_maps.append(
                go.Choroplethmap(
                    name='Subcatchments',
                    geojson=subcatchments_to_geo_crs.__geo_interface__,
                    locations=subcatchments_to_geo_crs.index,
                    z=[1] * len(subcatchments_to_geo_crs),
                    colorscale=[[0, 'salmon'], [1, 'salmon']],
                    hovertemplate='<br>'.join([
                        f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(subcatchments_to_geo_crs_data.columns[:10])
                    ]),
                    text=subcatchments_to_geo_crs['Name'],
                    customdata=subcatchments_to_geo_crs_data.values[:,:10],
                    showscale=False,
                    showlegend=True,
                    marker=dict(
                        opacity=0.3,
                    )
                )
            )

        return choropleth_maps

    @staticmethod
    def plot_links_plotly(swmm_model: SpatialSWMM, *args, **kwargs) -> List[go.Scattermap]:
        """
        This function plots the links of the SWMM model
        :param swmm_model:
        :param args:
        :param kwargs:
        :return:
        """
        link_traces: List[go.Scattermap] = []

        links_to_geo_crs = swmm_model.links.to_crs('EPSG:4326')
        links_to_geo_crs.insert(0, 'Name', links_to_geo_crs.index)

        conduits = links_to_geo_crs[links_to_geo_crs['LinkType'] == 'CONDUITS']
        conduits_data = pd.DataFrame(conduits.drop(columns='geometry'))

        orifices = links_to_geo_crs[links_to_geo_crs['LinkType'] == 'ORIFICES']
        orifices_data = pd.DataFrame(orifices.drop(columns='geometry'))

        weirs = links_to_geo_crs[links_to_geo_crs['LinkType'] == 'WEIRS']
        weirs_data = pd.DataFrame(weirs.drop(columns='geometry'))

        outlets = links_to_geo_crs[links_to_geo_crs['LinkType'] == 'OUTLETS']
        outlets_data = pd.DataFrame(outlets.drop(columns='geometry'))

        if not conduits.empty:
            first = True
            for index, row in conduits.iterrows():
                geometry = row['geometry']
                link_traces.append(
                    go.Scattermap(
                        name='Conduits',
                        lat=[coord[1] for coord in geometry.coords],
                        lon=[coord[0] for coord in geometry.coords],
                        mode='lines',
                        line=dict(
                            color='gray',
                            width=2
                        ),
                        hovertemplate='<br>'.join([
                            f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(conduits_data.columns[:10])
                        ]),
                        text=row['Name'],
                        customdata=conduits_data.loc[index].values[:10],
                        showlegend=first,
                        legendgroup='Conduits',
                        visible=first,
                    )
                )
                first = False

        if not orifices.empty:
            first = True
            for index, row in orifices.iterrows():
                geometry = row['geometry']
                link_traces.append(
                    go.Scattermap(
                        name='Orifices',
                        lat=[coord[1] for coord in geometry.coords],
                        lon=[coord[0] for coord in geometry.coords],
                        mode='lines',
                        line=dict(
                            color='#7e2a18',
                            width=4
                        ),
                        hovertemplate='<br>'.join([
                            f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(orifices_data.columns[:10])
                        ]),
                        text=row['Name'],
                        customdata=orifices_data.loc[index].values[:10],
                        showlegend=first,
                        legendgroup='Orifices'
                    )
                )
                first = False

        if not weirs.empty:
            first = True
            for index, row in weirs.iterrows():
                geometry = row['geometry']
                link_traces.append(
                    go.Scattermap(
                        name='Weirs',
                        lat=[coord[1] for coord in geometry.coords],
                        lon=[coord[0] for coord in geometry.coords],
                        mode='lines',
                        line=dict(
                            color='#18437e',
                            width=4.5
                        ),
                        hovertemplate='<br>'.join([
                            f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(weirs_data.columns[:10])
                        ]),
                        text=row['Name'],
                        customdata=weirs_data.loc[index].values[:10],
                        showlegend=first,
                        legendgroup='Weirs'
                    )
                )
                first = False

        if not outlets.empty:
            first = True
            for index, row in outlets.iterrows():
                geometry = row['geometry']
                link_traces.append(
                    go.Scattermap(
                        name='Outlets',
                        lat=[coord[1] for coord in geometry.coords],
                        lon=[coord[0] for coord in geometry.coords],
                        mode='lines',
                        line=dict(
                            color='#7e1843',
                            width=5
                        ),
                        hovertemplate='<br>'.join([
                            f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in enumerate(outlets_data.columns[:10])
                        ]),
                        text=row['Name'],
                        customdata=outlets_data.loc[index].values[:10],
                        showlegend=first,
                        legendgroup='Outlets'
                    )
                )
                first = False

        return link_traces

