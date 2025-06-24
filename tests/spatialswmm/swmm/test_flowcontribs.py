# python imports
import unittest

# 3rd party imports
import networkx as nx
import matplotlib.pyplot as plt

# project imports
from serto.swmm import *
from . import EXAMPLE_SWMM_TEST_MODEL_A


class TestFlowContributions(unittest.TestCase):

    def setUp(self):
        pass

    def test_downstream_flow(self):
        """
        Test the loading of a SWMM model
        :return:
        """
        model = SpatialSWMM(
            inp_file=EXAMPLE_SWMM_TEST_MODEL_A['filepath'],
            crs=EXAMPLE_SWMM_TEST_MODEL_A['crs']
        )

        # plot network x
        network = model.network

        # matplotlib networkx plot
        # nx.draw(network, with_labels=True, node_size=500, font_size=10, font_color='white')
        #
        # flow_array = swmm_flow_summary(
        #     swmm_instance=model,
        #     output_file=EXAMPLE_SWMM_TEST_MODEL_A['filepath'].replace('.inp', '.out')
        # )
        #
        # np.save(
        #     file=EXAMPLE_SWMM_TEST_MODEL_A['filepath'].replace('.inp', '.npy'),
        #     arr=flow_array
        # )

        flow_array = np.load(
            file=EXAMPLE_SWMM_TEST_MODEL_A['filepath'].replace('.inp', '.npy')
        )

        upstream_flow_frac, upstream_flows = node_flow_origins(
            flows=flow_array,
        )

        downstream_node = network.nodes['J8931.352']
        downstream_node_index = downstream_node['index']

        swmm_nodes = model.network.nodes
        swmm_nodes['J8931.352_upstream_flow_frac'] = upstream_flows[downstream_node_index]
        # swmm_nodes.



