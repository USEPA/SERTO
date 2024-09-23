# python imports
import os

# third-party imports

# local imports

TEST_DATA_DIR = os.path.abspath(os.path.dirname(__file__))
CVG_WIND_SPEED_DATA = os.path.join(TEST_DATA_DIR, 'CVGSpeed_dir.csv')
TEST_SWMM_INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_model.inp')
