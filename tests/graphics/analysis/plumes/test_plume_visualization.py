# python imports
import math
import unittest

# 3rd party imports
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# project imports
from ....analysis.wind import EXAMPLE_CVG_WIND_DATA

from serto.swmm import SpatialSWMM
from serto.analysis.wind import WindAnalysis
from serto.graphics.analysis.wind import WindVisualization
from serto.graphics.swmm import SpatialSWMMVisualization
from serto.analysis.plumes import GaussianPlume

from ....spatialswmm.swmm import EXAMPLE_SWMM_TEST_MODEL_A


class TestPlumeVisualization(unittest.TestCase):

    def setUp(self):
        self.wind_data = pd.read_csv(
            EXAMPLE_CVG_WIND_DATA,
            index_col="TimeEST",
            parse_dates=["TimeEST"]
        )

        self.model = SpatialSWMM.read_model(
            model_path=EXAMPLE_SWMM_TEST_MODEL_A['filepath'],
            crs=EXAMPLE_SWMM_TEST_MODEL_A['crs']
        )

        self.wind_data = self.wind_data.drop(columns=['station', 'station_name'])

        # remove duplicate indexes and sort
        self.wind_data = self.wind_data[~self.wind_data.index.duplicated(keep='first')]
        self.wind_data = self.wind_data.sort_index()

    def test_plume_visualization(self):
        """
        Test the plot_model function
        :return:
        """

        nodes_centroid = self.model.nodes.dissolve().centroid
        nodes_centroid_wgs = nodes_centroid.to_crs('EPSG:4326')

        easting = nodes_centroid.geometry.x[0]
        northing = nodes_centroid.geometry.y[0]

        lon = nodes_centroid_wgs.geometry.x[0]
        lat = nodes_centroid_wgs.geometry.y[0]

        subcatchments = SpatialSWMMVisualization.plot_catchments_plotly(self.model)
        wind_rose = WindVisualization.plot_wind_rose_bar_polar(
            wind_data=self.wind_data.resample(pd.Timedelta(hours=1)).mean(),
            wind_speed_col='sknt',
            wind_dir_col='drct',
            raxis_mode='fraction',
        )

        plume1 = GaussianPlume(
            source_strength=50 * 10 ** 6 / 100,
            source_location=(
                easting - 5000,
                northing - 2500,
            ),
            wind_speed=10,
            wind_direction=202.5,
            standard_deviation=(5000, 1000)
        )

        plume2 = GaussianPlume(
            source_strength=180 * 10 ** 6 / (4.0 * 100.0),
            source_location=(
                easting + 4000,
                northing + 1200,
            ),
            wind_speed=10,
            wind_direction=270,
            standard_deviation=(4000, 2000)
        )

        conc_label = "Cesium-137 (Micro Curies)"
        conc_label_flux = f"Cesium-137 (Micro Curies/ft^2)"

        plume1_sub, plum1 = plume1.area_weighted_buildup(
            mesh_resolution=100,
            sub_catchment=self.model.subcatchments,
            mass_loading_field=conc_label,
            mass_loadinf_flux_field=conc_label_flux
        )

        plume2_sub, plum2 = plume2.area_weighted_buildup(
            mesh_resolution=100,
            sub_catchment=self.model.subcatchments,
            mass_loading_field=conc_label,
            mass_loadinf_flux_field = conc_label_flux
        )

        min_flux = 0.5
        plum1 = plum1.to_crs('EPSG:4326')
        plume1_sub = plume1_sub.to_crs('EPSG:4326')
        plume1_valid = plum1[plum1[conc_label_flux] > min_flux]

        plum2 = plum2.to_crs('EPSG:4326')
        plume2_sub = plume2_sub.to_crs('EPSG:4326')
        plume2_valid = plum2[plum2[conc_label_flux] > min_flux]

        max_flux = max(
            plume1_valid[conc_label_flux].max(),
            plume2_valid[conc_label_flux].max(),
        )

        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [dict(type="polar"), None],
                [None, dict(type="map")],
            ],
            column_widths=[0.2, 0.8],
        )

        for bar in wind_rose:
            bar['legendgroup'] = 'wind_rose'
            fig.add_trace(bar, row=1, col=1)

        plume1_valid_data = pd.DataFrame(plume1_valid.drop(columns='geometry'))

        plume1_fig = go.Choroplethmap(
            name='Plume 1',
            geojson=plume1_valid.__geo_interface__,
            locations=plume1_valid.index,
            z=plume1_valid[conc_label_flux],
            colorscale='Jet',
            colorbar=dict(
                title=conc_label_flux,
                x=0.9,
            ),
            zmin=min_flux,
            zmax=max_flux,
            hovertemplate='<br>'.join([
                f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in
                enumerate(plume1_valid_data.columns)
            ]),
            customdata=plume1_valid_data.values,
            showlegend=True,
            marker=dict(
                opacity=0.5,
                line=dict(
                    width=0.0001
                )
            )
        )
        fig.add_trace(plume1_fig, row=2, col=2)

        plume2_valid_data = pd.DataFrame(plume2_valid.drop(columns='geometry'))

        plume2_fig = go.Choroplethmap(
            name='Plume 2',
            geojson=plume2_valid.__geo_interface__,
            locations=plume2_valid.index,
            z=plume2_valid[conc_label_flux],
            colorscale='Jet',
            colorbar=dict(
                title=conc_label_flux,
                x=0.9,
            ),
            showscale=False,
            zmin=min_flux,
            zmax=max_flux,
            hovertemplate='<br>'.join([
                f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in
                enumerate(plume2_valid_data.columns)
            ]),
            text=plume2_valid_data['label'],
            customdata=plume2_valid_data.values,
            showlegend=True,
            marker=dict(
                opacity=0.5,
                line=dict(
                    width=0.0001
                )
            )
        )
        fig.add_trace(plume2_fig, row=2, col=2)

        plume1_sub_data = pd.DataFrame(plume1_sub.drop(columns='geometry'))
        plumes_sub1_fig = go.Choroplethmap(
            name='Sub-catchment Flux Plume 1',
            geojson=plume1_sub.__geo_interface__,
            locations=plume1_sub.index,
            z=plume1_sub[conc_label_flux],
            colorscale='Jet',
            colorbar=dict(
                title=conc_label_flux,
                x=0.9,
            ),
            zmin=min_flux,
            zmax=max_flux,
            showscale=False,
            hovertemplate='<br>'.join([
                f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in
                enumerate(plume1_sub_data.columns)
            ]),
            customdata=plume1_sub_data.values,
            showlegend=True,
            marker=dict(
                opacity=0.5,
                line=dict(
                    width=0.0001
                )
            )
        )
        fig.add_trace(plumes_sub1_fig, row=2, col=2)

        plume2_sub_data = pd.DataFrame(plume2_sub.drop(columns='geometry'))
        plumes_sub2_fig = go.Choroplethmap(
            name='Sub-catchment Flux Plume 2',
            geojson=plume2_sub.__geo_interface__,
            locations=plume2_sub.index,
            z=plume2_sub[conc_label_flux],
            colorscale='Jet',
            colorbar=dict(
                title=conc_label_flux,
                x=0.9,
            ),
            zmin=min_flux,
            zmax=max_flux,
            showscale=False,
            hovertemplate='<br>'.join([
                f'<b>{col}</b>: %{{customdata[{i}]}}' for i, col in
                enumerate(plume2_sub_data.columns)
            ]),
            customdata=plume2_sub_data.values,
            showlegend=True,
            marker=dict(
                opacity=0.5,
                line=dict(
                    width=0.0001
                )
            )
        )
        fig.add_trace(plumes_sub2_fig, row=2, col=2)

        fig.update_layout(
            title=f'',
            map_style="carto-positron",
            showlegend=True,
            map={
                'center': {'lon': lon, 'lat': lat},
                'zoom': 13,
                'domain': dict(
                    x=[0.2, 0.9],
                    y=[0.0, 1.0]
                )
            },
            legend_tracegroupgap=30,
        )

        plotly.offline.plot(fig, filename='test_plume_visualization.html')
