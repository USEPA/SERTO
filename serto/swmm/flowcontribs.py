import numpy as np
from jinja2.nodes import Output
from numpy.typing import NDArray
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, identity, lil_matrix
from scipy.sparse.linalg import inv, spsolve
from . import SpatialSWMM
from epaswmm import output

def node_flow_destinations(
    flows: csr_matrix
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the flows and fractional flows at downstream nodes that eventually receive flows from upstream nodes.

    :param flows: (numpy.ndarray or scipy.sparse.csr_matrix): nxn matrix representing flows between nodes.
    :returns: numpy.ndarray: nxn matrix where each row represents the upstream node and columns
    represent the flows and fractional flows.
    """

    n = flows.shape[0]

    # Create an identity matrix as a sparse matrix
    identity_matrix = identity(n, format='csr')

    # Transpose the flow matrix to consider the perspective of receiving nodes
    transposed_flow_matrix = flows.transpose()

    # Solve the linear system (I - transposed_flow_matrix) * X = I directly
    flow_fractions = spsolve(identity_matrix - transposed_flow_matrix, identity_matrix)

    # Calculate the flow magnitudes by multiplying the flow fractions by the transposed flow matrix
    flow_magnitudes = flow_fractions @ transposed_flow_matrix.toarray()

    return flow_fractions, flow_magnitudes


def node_flow_origins(flows: csr_matrix):
    """
    Calculate the fraction and magnitudes of flows from upstream nodes to downstream nodes.
    :param flows: (numpy.ndarray or scipy.sparse.csr_matrix): nxn matrix representing flows between nodes.
    :return: numpy.ndarray: nxn matrix where each row represents the downstream node
    and columns represent the flows and fractions.
    """
    n = flows.shape[0]

    # Create an identity matrix as a sparse matrix
    identity_matrix = identity(n, format='csr')

    # Solve the linear system (I - flow_matrix) * X = I directly
    flow_fractions = spsolve(identity_matrix - flows, identity_matrix)

    # Calculate the flow magnitudes by multiplying the flow fractions by the original flow matrix
    flow_magnitudes = flow_fractions @ flows.toarray()

    return flow_fractions, flow_magnitudes


def flip_edge(g, u, v):
    """
    Flip the direction of an edge in a directed graph.
    :param G: The directed graph.
    :param u: The source node of the edge.
    :param v: The target node of the edge.
    :return: None
    """
    attr = g.get_edge_data(u, v)
    g.remove_edge(u, v)
    g.add_edge(v, u, **attr)

def swmm_flow_summary(
        swmm_instance: SpatialSWMM,
        output_file: str
) :
    """
    Calculate flow contributions in a SWMM model.
    #TODO Implement sparse matrix for flow contributions
    :param swmm_instance:
    :param output_file:
    :return:
    """

    network = swmm_instance.network

    n = len(network.nodes)
    flow_array = lil_matrix((n, n), dtype=np.float64)

    with output.Output(output_file) as out:

        edges_to_flip = []

        for node, node_attr in network.nodes(data=True):
            for u, v, edge in network.out_edges(node, data=True):
                edge_name = edge['name']
                edge_flows = out.get_link_timeseries(
                   element_index=edge_name,
                   attribute = output.LinkAttribute.FLOW_RATE
                )
                edge_flow_vals = np.array(list(edge_flows.values()))

                # max_val = edge_flow_vals.max()
                sum_flow = edge_flow_vals.sum()

                if sum_flow >= 0:
                    u_index = node_attr['index']
                    v_index = network.nodes[v]['index']
                    flow_array[u_index, v_index] += sum_flow
                else:
                    edges_to_flip.append((u, v, node_attr, sum_flow))


        for u, v, node_attr, sum_flow in edges_to_flip:
            flip_edge(network, u, v)
            u_index = network.nodes[v]['index']
            v_index = node_attr['index']
            flow_array[u_index, v_index] += abs(sum_flow)

    return  flow_array.tocsr()

def swmm_network_csr_matrix(
        swmm_instance: SpatialSWMM
) -> csr_matrix:
    """
    Create a connectivity matrix for the SWMM network.
    :param swmm_instance: The SWMM model instance.
    :return: A sparse connectivity matrix.
    """
    network = swmm_instance.network
    n = len(network.nodes)
    connectivity = csr_matrix((n, n), dtype=np.float64)

    for u, v in network.edges():
        u_index = network.nodes[u]['index']
        v_index = network.nodes[v]['index']
        connectivity[u_index, v_index] = 1.0

    return connectivity