# python imports
import unittest

# third-party imports
import pandas as pd
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('TkAgg')

# local imports
from swmmoptimizer.tests.data import CVG_WIND_SPEED_DATA


class TestWindAnalysis(unittest.TestCase):
    """
    This class tests the wind analysis module
    """

    def setUp(self):
        """
        This function sets up the test
        """
        self.data = pd.read_csv(
            CVG_WIND_SPEED_DATA,
            index_col="TimeEST",
            parse_dates=["TimeEST"]
        )

        self.data['sknt'] *= 0.514444

    def test_wind_analysis(self):
        """
        This function tests the wind analysis module
        """
        ax = WindroseAxes.from_ax()
        ax.box(self.data['drct'], self.data['sknt'], normed=True, edgecolor='white', cmap=cm.plasma)
        ax.box(self.data['drct'], self.data['sknt'],edgecolor='white')

        ax.set_legend()