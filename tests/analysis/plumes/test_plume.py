# python imports
import unittest

import matplotlib.pyplot as plt
import numpy as np

# third-party imports

# local imports
from serto.analysis.plumes import GaussianPlume
from serto.swmm import SpatialSWMM

from tests.spatialswmm.swmm import EXAMPLE_SWMM_TEST_MODEL_A

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

    def test_2d_gaussian_with_standard_deviation(self):
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
            source_location=(5267829.351724993, 4264426.024403716),
            wind_direction=275,
            standard_deviation=(5000, 1000),
            exponential_decay_rate=0.00025
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

    def test_export_2d_gaussian_with_standard_deviation(self):
        """
        
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
            source_strength=1200.0,
            source_location=(5267829.351724993, 4264426.024403716),
            wind_direction=275,
            standard_deviation=(5000, 1000),
            exponential_decay_rate=0.00025
        )

        self.model = SpatialSWMM.read_model(
            model_path=EXAMPLE_SWMM_TEST_MODEL_A['filepath'],
            crs=EXAMPLE_SWMM_TEST_MODEL_A['crs']
        )

        conc_label = "Cesium-137 (Curies)"
        conc_label_flux = f"Cesium-137 (Curies/ft^2)"

        plume_subcatchments, plum_mesh = plume.area_weighted_buildup(
            mesh_resolution=100,
            sub_catchment=self.model.subcatchments,
            mass_loading_field=conc_label,
            mass_loadinf_flux_field=conc_label_flux
        )

        plume_subcatchments.to_file("plume_subcatchments.shp")
        plum_mesh.to_file("plume_mesh.shp")


    def test_2d_gaussian_with_wind_plume(self):
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
            source_location=(5267829.351724993, 4264426.024403716),
            wind_direction=275,
            wind_speed=6,
            stability_coefficient=0.06,
            stability_exponent=0.92
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
        matplotlib.use('TkAgg')

    def tearDown(self):
        """
        This function tears down the test
        """
        pass
