# # python imports
# import unittest
# from datetime import datetime

# import matplotlib.pyplot as plt
# import numpy as np

# # third-party imports

# # local imports
# from serto.analysis.plumes import GaussianPlume, PlumeEventMatrix
# from serto.graphics.swmm.spatialswmm import SpatialSWMMVisualization

# from ..swmm.swmm import EXAMPLE_SWMM_TEST_MODEL_A


# class TestPlumeEventMatrix(unittest.TestCase):
#     """
#     This class tests the PlumeEventMatrix module
#     """

#     def setUp(self):
#         """
#         This function sets up the test
#         """
#         pass

#     def test_plume_event_matrix(self):
#         """
#         This function tests the PlumeEventMatrix module
#         """
#         plume_event_matrix = PlumeEventMatrix(
#             swmm_input_file=EXAMPLE_SWMM_TEST_MODEL_A['filepath'],
#             crs=EXAMPLE_SWMM_TEST_MODEL_A['crs'],
#             contaminant_loading_spatial_resolution=250,
#             release_time=datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'),
#             contaminant_name='Cesium',
#             contaminant_units='Curies',
#             plume_type=GaussianPlume.PlumeType.EMPIRICAL,
#             contaminant_amounts=[266.0e12, 266.0e12],
#             wind_directions=45,
#             wind_speeds=1.0,
#             standard_deviations=[
#                 (2000, 50),
#                 (2200, 100),
#                 (2400, 150)
#             ],
#             release_location_element_types=['STORAGES', ],
#             release_locations={'Example_Location': (
#                 5263810.299,
#                 4272675.393
#             )},
#             release_grid=(1000, 1000)
#         )

#         plume_event_summaries = plume_event_matrix.plume_events_summary
#         plume_event_summaries.head()

#         import plotly

#         figs = SpatialSWMMVisualization.plot_plume_event_matrix(plume_event_matrix, event_indexes=[0, 1, 2])

#         plotly.offline.plot(figs[0], filename='plume_event_matrix.html')


