import os
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SWMM_TEST_MODEL_A = {
    'filepath': os.path.join(HERE, 'test_model.inp'),
    'crs': 'EPSG:3089',
}