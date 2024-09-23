# python imports
import unittest

# 3rd party imports

# project imports
from ...spatialswmm import SpatialSWMM
from swmmoptimizer.tests.data import TEST_SWMM_INPUT_FILE


class TestSpatialSWMM(unittest.TestCase):

    def setUp(self):
        pass

    def test_swmm_load(self):
        model = SpatialSWMM.read_model(TEST_SWMM_INPUT_FILE, crs='epsg:3725')

        title = ['Banklick Creek (Downstream) SWMM model containing surveyed open channel sections, bridges from '
                 'previous HEC-RAS model, and surveyed pipes and manholes for planning level watershed assessment.  '
                 'Subcatchments delineated (5/15/2017) using KY DEM and aerial imagery.  CN based on NLCD 2011 and '
                 'SSURGO data.']

        self.assertEqual(model.title, title)
