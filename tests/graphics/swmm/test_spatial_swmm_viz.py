# python imports
import unittest

# 3rd party imports

# project imports
from serto.swmm import SpatialSWMM
from serto.graphics.swmm import SpatialSWMMVisualization
from tests.spatialswmm.swmm import EXAMPLE_SWMM_TEST_MODEL_A


class TestSpatialSWMMVizualization(unittest.TestCase):

    def setUp(self):
        self.model = SpatialSWMM.read_model(
            model_path=EXAMPLE_SWMM_TEST_MODEL_A['filepath'],
            crs=EXAMPLE_SWMM_TEST_MODEL_A['crs']
        )

    def test_plot_model(self):
        """
        Test the plot_model function
        :return:
        """
        node_traces = SpatialSWMMVisualization.plot_nodes_plotly(self.model)

        self.assertEqual(len(node_traces), 1)
