# python imports
from typing import List, Dict, Union, Tuple
import unittest

import pandas as pd

# third-party imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# project imports
from serto.analysis.wind import WindAnalysis
from . import EXAMPLE_CVG_WIND_DATA

class TestWindAnalysis(unittest.TestCase):
    """
    This class tests the wind analysis module
    """

    def setUp(self):
        """
        This function sets up the test
        """
        self.wind_data = pd.read_csv(
            EXAMPLE_CVG_WIND_DATA,
            index_col="TimeEST",
            parse_dates=["TimeEST"]
        )

        # Convert from knots to m/s
        self.wind_data['sknt'] = self.wind_data['sknt'] * 0.514444

    def test_speed_direction_joint(self):
        """
        This function tests the wind analysis module
        """
        model = WindAnalysis.joint_speed_direction_gmm_model(
            wind_data=self.wind_data,
            wind_speed_col='sknt',
            wind_dir_col='drct',
            gmm_type='GaussianMixture',
            n_components=16,
            covariance_type='full'
        )

        self.wind_data['cluster'] = model.predict(self.wind_data[['sknt', 'drct']])
        self.wind_data['likelihood'] = model.score_samples(self.wind_data[['sknt', 'drct']])

        print(self.wind_data)