# python imports
import unittest

import matplotlib.pyplot as plt
import numpy as np

# third-party imports

# local imports
from ....analysis.plumes import GaussianPlume


class TestGaussianPlume(unittest.TestCase):
    """
    This class tests the Gaussian plume module
    """

    def setUp(self):
        """
        This function sets up the test
        """

        self.x = np.arange(
            start=-5000,
            stop=5000,
            step=10
        )

        self.y = np.arange(
            start=-5000,
            stop=5000,
            step=10
        )

        self.z = np.arange(
            start=0,
            stop=100,
            step=10
        )

        self.locations_2d = np.array(np.meshgrid(self.x, self.y)).T.reshape(-1, 2)
        self.locations_3d = np.array(np.meshgrid(self.x, self.y, self.z)).T.reshape(-1, 3)

    def test_2d_gaussian_with_standard_deviation_plume(self):
        """
        This function tests the Gaussian plume module
        """

        plume = GaussianPlume(
            source_strength=100.0,
            source_location=(0.0, 0.0),
            direction=145,
            standard_deviation=(2500, 833)
        )

        concentrations = plume.concentration(self.locations_2d)
        mesh_grid_concs = concentrations.reshape((self.x.shape[0], self.y.shape[0]))
        fig, ax = plt.subplots()
        cb = ax.contourf(self.x, self.y, mesh_grid_concs)
        fig.colorbar(cb, label="Concentration (Picocuries)")

    def test_2d_gaussian_with_wind_plume(self):
        """
        This function tests the Gaussian plume module
        """

        plume = GaussianPlume(
            source_strength=100.0,
            source_location=(0.0, 0.0),
            direction=145,
            wind_speed=30,
        )

        concentrations = plume.concentration(self.locations_2d * 0.3048)
        mesh_grid_concs = concentrations.reshape((self.x.shape[0], self.y.shape[0]))
        fig, ax = plt.subplots()
        cb = ax.contourf(self.x, self.y, mesh_grid_concs)
        fig.colorbar(cb, label="Concentration (Picocuries)")

    def tearDown(self):
        """
        This function tears down the test
        """
        pass
