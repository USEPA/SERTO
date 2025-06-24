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
    For each node, estimate the absolute and fractional flows from that node
    that reach every other node in the network.
    :param flows: csr_matrix, shape (n, n), flows[i, j] = flow from i to j
    :return: (absolute_flows, fractional_flows), both shape (n, n)
             absolute_flows[i, j]: flow from i that ends up at j
             fractional_flows[i, j]: fraction of i's outflow that ends up at j
    """
    n = flows.shape[0]
    outflows = np.array(flows.sum(axis=1)).flatten()
    outflows[outflows == 0] = 1  # Avoid division by zero

    # Transition matrix: T[i, j] = fraction of i's outflow going to j
    T = flows.multiply(1 / outflows[:, None])

    # Fundamental matrix: (I - T)^-1
    I = identity(n, format='csr')
    F = spsolve(I - T, I)  # shape (n, n)
    F = np.array(F).reshape((n, n))

    # F[i, j]: fraction of flow at node j that originated from node i
    # For destinations, we want: for each i, the amount from i that ends up at j
    # So, absolute_flows[i, j] = F[i, j] * outflows[i]
    absolute_flows = F * outflows[:, None]

    # Fractional flows: for each i, fraction of i's outflow that ends up at j
    row_sums = absolute_flows.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    fractional_flows = absolute_flows / row_sums[:, None]

    return absolute_flows, fractional_flows


def node_flow_origins(flows: csr_matrix):
    """
    Given a flow matrix (i->j), compute for each downstream node:
    1. The absolute flow at the downstream node that originated from each upstream node.
    2. The fraction of the downstream node's total flow that originated from each upstream node.
    :param flows: csr_matrix, shape (n, n), flow_matrix[i, j] = flow from i to j
    :return: (absolute_contributions, fractional_contributions), both shape (n, n)
    """
    n = flows.shape[0]
    # Compute total outflow from each node
    outflows = np.array(flows.sum(axis=1)).flatten()
    outflows[outflows == 0] = 1  # Avoid division by zero

    # Transition matrix: T[i, j] = fraction of i's outflow going to j
    T = flows.multiply(1 / outflows[:, None])

    # Fundamental matrix: (I - T)^-1
    I = identity(n, format='csr')
    F = spsolve(I - T, I.toarray())  # shape (n, n)

    # F[i, j]: fraction of flow at node j that originated from node i
    F = np.array(F).reshape((n, n))

    # Compute total inflow at each node
    inflows = np.array(flows.sum(axis=0)).flatten()

    # Absolute contributions: for each downstream node j, flow from i = F[i, j] * inflow at j
    absolute_contributions = F * inflows[None, :]

    # Fractional contributions: for each downstream node j, fraction from i = F[i, j] / sum_k F[k, j]
    col_sums = absolute_contributions.sum(axis=0)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    fractional_contributions = absolute_contributions / col_sums[None, :]

    return absolute_contributions, fractional_contributions


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
                    # flow_array[u_index, v_index, 0] += max_val
                    flow_array[u_index, v_index] += sum_flow
                else:
                    edges_to_flip.append((u, v, node_attr, sum_flow))
                    # save flow values in reverse direction
                    # u_index = network.nodes[v]['index']
                    # v_index = node_attr['index']
                    # flow_array[u_index, v_index, 0] += max_val
                    # flow_array[u_index, v_index, 1] += abs(sum_flow)

        for u, v, node_attr, sum_flow in edges_to_flip:
            flip_edge(network, u, v)
            u_index = network.nodes[v]['index']
            v_index = node_attr['index']
            # flow_array[u_index, v_index, 0] += max_val
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