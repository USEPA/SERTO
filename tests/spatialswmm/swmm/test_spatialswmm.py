# python imports
import unittest

# 3rd party imports

# project imports
from serto.swmm import SpatialSWMM
from . import EXAMPLE_SWMM_TEST_MODEL_A


class TestSpatialSWMM(unittest.TestCase):

    def setUp(self):
        pass

    def test_swmm_load(self):
        """
        Test the loading of a SWMM model
        :return:
        """
        model = SpatialSWMM.read_model(
            model_path=EXAMPLE_SWMM_TEST_MODEL_A['filepath'],
            crs=EXAMPLE_SWMM_TEST_MODEL_A['crs']
        )

        title = ['Banklick Creek (Downstream) SWMM model containing surveyed open channel sections, bridges from '
                 'previous HEC-RAS model, and surveyed pipes and manholes for planning level watershed assessment.  '
                 'Subcatchments delineated (5/15/2017) using KY DEM and aerial imagery.  CN based on NLCD 2011 and '
                 'SSURGO data.']

        self.assertEqual(model.title, title)
