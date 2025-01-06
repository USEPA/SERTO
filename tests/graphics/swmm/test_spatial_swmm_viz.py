# python imports
import unittest

# 3rd party imports

# project imports
from serto.swmm import SpatialSWMM
from serto.graphics.swmm import (
    plot_swmm_nodes
)
from tests.data.swmm import TEST_SWMM_INPUT_FILE


class TestSpatialSWMMVizualization(unittest.TestCase):

    def setUp(self):
        self.model = SpatialSWMM.read_model(TEST_SWMM_INPUT_FILE, crs='EPSG:3089')

    def test_plot_model(self):
        """
        Test the plot_model function
        :return:
        """
        node_traces = plot_swmm_nodes(self.model)

        self.assertEqual(len(node_traces), 1)
