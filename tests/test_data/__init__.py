import os

HERE = os.path.dirname(os.path.abspath(__file__))

LIKING_RIVER_FLOW_DATA = os.path.join(HERE, 'LIKING_RIVER', 'liking_river_flow_03254520.csv')

EXAMPLE_SWMM_TEST_MODEL_A = {
    'filepath': os.path.join(HERE, 'LIKING_RIVER', 'test_model.inp'),
    'crs': 'EPSG:3089',
}