# python imports
import unittest

import matplotlib.pyplot as plt
import numpy as np

# third-party imports

# local imports
from serto.analysis.plumes import GaussianPlume


class TestPlume(unittest.TestCase):
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

        self.locations_2d = np.array(np.meshgrid(self.x, self.y)).T.reshape(-1, 2)

    def test_2d_gaussian_with_standard_deviation_plume(self):
        """
        This function tests the Gaussian plume module
        """
        x = np.arange(
            start=5255271.338,
            stop=5279182.901,
            step=100.0
        )

        y = np.arange(
            start=4250587.132,
            stop=4276449.389,
            step=100.0
        )

        xx, yy = np.meshgrid(x, y)

        locations_2d = np.array([xx.ravel(), yy.ravel()]).T

        plume = GaussianPlume(
            source_strength=1000000.0,
            source_location=(5267829.351724993 + 5000, 4264426.024403716 + 5000),
            wind_direction=90,
            standard_deviation=(5000, 1000)
        )

        import matplotlib
        matplotlib.use('TkAgg')

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        concentrations = plume.concentration(locations_2d)
        mesh_grid_concs = concentrations.reshape((y.shape[0], x.shape[0]))

        cb = ax.contourf(x, y, mesh_grid_concs)
        fig.colorbar(cb, label="Concentration (Picocuries)")

        matplotlib.use('TkAgg')
        plt.show()
        print("Concentrations: ", concentrations)

    def test_2d_gaussian_with_wind_plume(self):
        """
        This function tests the Gaussian plume module
        """

        plume = GaussianPlume(
            source_strength=1000000.0,
            source_location=(0.0, 0.0),
            direction=45,
            wind_speed=5,
            turbulent_intensity=0.55
        )

        import matplotlib
        matplotlib.use('TkAgg')

        from matplotlib import pyplot as plt

        concentrations = plume.concentration(self.locations_2d)
        mesh_grid_concs = concentrations.reshape((self.x.shape[0], self.y.shape[0]))
        fig, ax = plt.subplots()
        cb = ax.contourf(self.x, self.y, mesh_grid_concs)
        fig.colorbar(cb, label="Concentration (Picocuries)")

        plt.show()
        matplotlib.use('TkAgg')
        print("Concentrations: ", concentrations)

    def tearDown(self):
        """
        This function tears down the test
        """
        pass
