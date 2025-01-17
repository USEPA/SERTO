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
        lon = nodes_centroid_wgs.geometry.x[0]
        lat = nodes_centroid_wgs.geometry.y[0]

        subcatchments = SpatialSWMMVisualization.plot_catchments_plotly(self.model)
        wind_rose = WindVisualization.plot_wind_rose_bar_polar(
            wind_data=self.wind_data.resample(pd.Timedelta(hours=1)).mean(),
            wind_speed_col='sknt',
            wind_dir_col='drct',
            raxis_mode='fraction',
        )

        plume = GaussianPlume(
            source_strength=10000000,
            source_location= [
                nodes_centroid.geometry.x[0],
                nodes_centroid.geometry.y[0],
            ],
            wind_speed=10,
            wind_direction=202.5,
            standard_deviation=[5000, 500]
        )

        subcatchments_bounds = self.model.subcatchments.total_bounds

        subcatchment_with_buildup, get_subcatchment_plum = plume.area_weighted_buildup(
            sub_catchment=self.model.subcatchments,
        )

        get_subcatchment_plum = get_subcatchment_plum.to_crs('EPSG:4326')
        valid_concentrations = get_subcatchment_plum[get_subcatchment_plum['concentration'] > 1.0e-3]

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

        fig.add_traces(subcatchments, rows=[2], cols=[2])

        plume_fig = go.Choroplethmap(
            name='Plume',
            geojson=valid_concentrations.__geo_interface__,
            locations=valid_concentrations.index,
            z=valid_concentrations['concentration'],
            colorscale='Viridis',
            colorbar=dict(
                title='Concentration (Picocuries)',
                x=0.9,
            ),
            hoverinfo='location+z',

            showlegend=True,
            marker=dict(
                opacity=0.5,
                line=dict(
                    width=0.0001
                )
            )
        )

        fig.add_trace(plume_fig, row=2, col=2)

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
            # legend title

            # colorbar=dict(
            #     len=0.5,
            #     y=0.5,
            #     x=0.0
            # ),
            legend_tracegroupgap=30,
        )

        plotly.offline.plot(fig, filename='test_plume_visualization.html')
