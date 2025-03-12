"""
sero.analysis.plumes.GaussianPlume contains the functionality for generating a Gaussian plume.
"""

# python imports
from typing import Tuple, List, Any, Union, Dict
from enum import Enum
from datetime import datetime
import math

# third-party imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import geopandas as gpd
from shapely.geometry import Polygon

# local imports
from ... import IDictable
from ...swmm import SpatialSWMM

class GaussianPlume(IDictable):
    """
    This class generates a Gaussian plume
    """

    class PlumeType(Enum):
        """
        This class defines the plume type
        """
        EMPIRICAL_LINEAR = 1
        EMPIRICAL_EXPONENTIAL = 2
        PHYSICS_BASED = 3

    def __init__(
            self,
            source_strength: float,
            source_location: Tuple[float, float],
            wind_direction: float,
            standard_deviation: Tuple[float, float] = None,
            exponential_decay_rate: float = None,
            wind_speed: float = 10.0,
            stability_coefficient: float = 0.06,
            stability_exponent: float = 0.92,
            *args,
            **kwargs

    ) -> None:
        """
        This function initializes the plume generation object.
        All units are SI units. If standard deviation is not provided the
        physics-based formulation is used. If standard deviation is provided
        the empirical formulation is used.

        :param source_strength: Source strength flux in units of mass per unit area
        :param source_location: An array of the form [x, y]
          specifying the location of the source in meters
        :param wind_direction: Angle in degrees for direction of plume in degrees
          from the x-axis
        :param standard_deviation: An array of the form [downwind, crosswind]
          specifying the extent of the plume
        :param standard_deviation_exp_growth_rate: Exponential growth rate of the standard deviation in m/s
        :param exponential_decay: Exponential decay factor
        :param wind_speed : Wind speed in m/s
        :param turbulent_intensity : Lateral turbulent intensity in m/s
        """

        self.source_strength = source_strength
        self.source_location = source_location
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed

        self.stability_coefficient = stability_coefficient
        self.stability_exponent = stability_exponent

        self.standard_deviation = standard_deviation
        self.exponential_decay_rate = exponential_decay_rate

        self._direction_vector = np.squeeze(np.array([
            math.cos(math.radians(self.wind_direction)),
            math.sin(math.radians(self.wind_direction))
        ]))

        if standard_deviation is None:
            self._plume_type = self.PlumeType.PHYSICS_BASED
        else:
            if self.exponential_decay_rate == 0.0 or self.exponential_decay_rate is None:
                self._plume_type = self.PlumeType.EMPIRICAL_LINEAR
            else:
                self._plume_type = self.PlumeType.EMPIRICAL_EXPONENTIAL
    @property
    def plume_type(self):
        """
        This function returns the plume type (empirical or physics-based)
        :return: The plume type (empirical or physics-based)
        """
        return self._plume_type

    def overlaps(self, locations: np.array) -> np.array:
        """
    This function returns True if the location is inside the plume and False
    otherwise
    : param locations: An array of the form [x, y]
    :returns: True if the location is inside the plume and False otherwise
    """

        concs = self.concentration(locations)
        perc = concs * 100 / self.source_strength

        return perc > 1.0e-12

    def concentration(self, locations: np.array) -> np.array:
        """
        This function returns the concentration at the location
        : param locations: An array of the form [x, y]
        :returns: The concentration at the location
        """
        distances = locations[..., :2] - np.array(self.source_location)
        distances = np.linalg.norm(distances, axis=1)

        downwind, crosswind, radis = self._proj_vectors(locations[..., :2])
        d = np.dot(downwind, self._direction_vector)

        concentration = np.zeros(locations.shape[0])

        valid_mesh_filter = d >= 0.0
        concentration[~valid_mesh_filter] = 0.0

        downwind_v = downwind[valid_mesh_filter]
        crosswind_v = crosswind[valid_mesh_filter]

        downwind_norm = np.linalg.norm(downwind_v, axis=1)
        crosswind_norm = np.linalg.norm(crosswind_v, axis=1)

        if self.plume_type == self.PlumeType.PHYSICS_BASED:
            # physics-based formulation

            downwind_standard_dev = self.stability_coefficient *  np.power(downwind_norm, self.stability_exponent)
            crosswind_standard_dev = self.stability_coefficient * np.power(crosswind_norm, self.stability_exponent)

            source_strength = self.source_strength / (
                    2.0 * np.pi * self.wind_speed * downwind_standard_dev * crosswind_standard_dev
            )
        elif self.plume_type == self.PlumeType.EMPIRICAL_LINEAR:

            downwind_standard_dev = self.standard_deviation[0]
            crosswind_standard_dev = self.standard_deviation[1] * downwind_norm / self.standard_deviation[0]
            source_strength = self.source_strength

        else:
            downwind_standard_dev = self.standard_deviation[0]

            crosswind_standard_dev = self.standard_deviation[1] * (
                    1.0 - np.exp(-self.exponential_decay_rate * downwind_norm)
            )

            source_strength = self.source_strength

        valid_concentrations = source_strength * np.exp(
            -0.5 * (crosswind_norm / crosswind_standard_dev) ** 2.0
        ) * np.exp(
            -0.5 * (downwind_norm / downwind_standard_dev) ** 2.0
        )

        concentration[valid_mesh_filter] = valid_concentrations
        # concentration[distances <= 0.0] = self.source_strength

        # import matplotlib
        # matplotlib.use('TkAgg')
        #
        # from matplotlib import pyplot as plt
        #
        # fig, ax = plt.subplots(2,2)
        # mesh_grid_concs = concentration.reshape((1000, 1000))
        #
        # x = np.arange(
        #     start=-5000,
        #     stop=5000,
        #     step=10
        # )
        #
        # y = np.arange(
        #     start=-5000,
        #     stop=5000,
        #     step=10
        # )
        #
        # cb = ax[0, 0].contourf(x, y, mesh_grid_concs)
        # fig.colorbar(cb, label="Concentration (Picocuries)")
        #
        # concentration[d > 0] = downwind_standard_dev
        # cb = ax[0, 1].contourf(x, y, concentration.reshape((1000, 1000)))
        # fig.colorbar(cb, label="Downwind (Picocuries)")
        #
        # concentration[d > 0] = crosswind_standard_dev
        # cb = ax[1,1].contourf(x, y, concentration.reshape((1000, 1000)))
        # fig.colorbar(cb, label="Crosswind (Picocuries)")
        #
        # matplotlib.use('TkAgg')
        # plt.show()

        return concentration

    def _proj_vectors(self, locations: np.array) -> np.array:
        """
        This function returns the distance from the source
        :param locations: An array of the form [x, y]
          specifying the location of the source
        :return: The distance from the source. Distance is valid
          if positive and invalid if <= 0
        """
        r = locations - np.array(self.source_location)
        a1 = np.dot(r, np.squeeze(self._direction_vector))[:, None] * self._direction_vector
        a2 = r - a1
        d = np.linalg.norm(r, axis=1)

        return a1, a2, d

    def plume_mesh(
            self,
            x_min:float ,
            x_max:float ,
            y_min:float ,
            y_max:float ,
            resolution:float,
            crs: str,
            mass_loading_field: str = "Concentration",
            mass_loading_flux_field: str = "Concentration (Mass/Area)"
    ) -> gpd.GeoDataFrame:
        """
        This function returns the concentrations for each point in a square polygon

        :param x_min: The minimum x location of the domain
        :param x_max: The maximum x location of the domain
        :param y_min: The minimum y location of the domain
        :param y_max: The maximum y location of the domain
        :param resolution: The resolution of the grid
        :param crs: The coordinate reference system of the grid
        :param mass_loading_field: The name of the mass field in the GeoDataFrame
        :param mass_loading_flux_field: The name of the mass per unit area field in the GeoDataFrame
        :return: A GeoDataFrame with the concentration field
        """

        half_resolution = resolution / 2.0
        x_pts = np.arange(x_min + half_resolution, x_max, resolution)
        y_pts = np.arange(y_min + half_resolution, y_max, resolution)

        x_pts_index = np.arange(x_pts.shape[0])
        y_pts_index = np.arange(y_pts.shape[0])

        x_pts, y_pts = np.meshgrid(x_pts, y_pts)
        x_pts_index, y_pts_index = np.meshgrid(x_pts_index, y_pts_index)

        locations = np.array([x_pts.ravel(), y_pts.ravel()]).T
        concentrations = self.concentration(locations)

        concentration_polygons = gpd.GeoDataFrame(
            {
                "geometry":  [
                    Polygon([
                        (x - half_resolution, y - half_resolution),
                        (x + half_resolution, y - half_resolution),
                        (x + half_resolution, y + half_resolution),
                        (x - half_resolution, y + half_resolution)
                    ])
                    for x, y in locations
                ],
                mass_loading_flux_field: concentrations.flatten(),
                "row_index": x_pts_index.flatten(),
                "col_index": y_pts_index.flatten(),
                'label': [f'{y}_{x}' for x, y in zip(x_pts_index.flatten(), y_pts_index.flatten())]
            },
            crs=crs
        )

        concentration_polygons[mass_loading_field] = \
            concentration_polygons[mass_loading_flux_field] * concentration_polygons.area

        return concentration_polygons

    def area_weighted_buildup(
            self, sub_catchment: gpd.GeoDataFrame,
            mesh_resolution: float = 100.0,
            mass_loading_field: str = "Concentration",
            mass_loadinf_flux_field: str = "Concentration (Mass/Area)"
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        This function calculates the area weighted buildup
        :param sub_catchment: The sub-catchment
        :param concentration_field: The concentration field
        :param buildup_field: The buildup field
        :param mesh_resolution: The mesh resolution
        :returns
        """

        subcatchment_fluxes = sub_catchment.copy()
        subcatchment_fluxes['name'] = subcatchment_fluxes.index
        subcatchment_fluxes['computed_area'] = subcatchment_fluxes.area
        subcatchment_fluxes[mass_loading_field] = 0.0
        subcatchment_fluxes[mass_loadinf_flux_field] = 0.0

        subcatchments_bounds = sub_catchment.total_bounds
        concentration_mesh = self.plume_mesh(
            x_min=subcatchments_bounds[0],
            y_min=subcatchments_bounds[1],
            x_max=subcatchments_bounds[2],
            y_max=subcatchments_bounds[3],
            resolution=mesh_resolution,
            crs=sub_catchment.crs,
            mass_loading_field=mass_loading_field,
            mass_loading_flux_field=mass_loadinf_flux_field
        )

        temp_name = f'{mass_loading_field}_p'
        temp_name_f = f'{mass_loadinf_flux_field}_p'

        concentration_mesh.rename(columns={
            mass_loading_field: temp_name,
            f'{mass_loading_field} (Mass/Area)': temp_name_f
        }, inplace=True)

        concentration_mesh['mesh_computed_area'] = concentration_mesh.area

        overlayed = gpd.overlay(concentration_mesh, subcatchment_fluxes, keep_geom_type=True, how='union')
        overlayed['overlayed_computed_area'] = overlayed.area

        for group_id, group_cells in overlayed.groupby('name'):
            total_mass = group_cells['overlayed_computed_area'] * group_cells[temp_name] / group_cells["mesh_computed_area"]
            subcatchment_fluxes.loc[group_id, mass_loading_field] = total_mass.sum()
            subcatchment_fluxes.loc[group_id, mass_loadinf_flux_field] = \
                total_mass.sum() /subcatchment_fluxes.loc[group_id, 'computed_area']


        concentration_mesh.rename(
            columns={
                temp_name: mass_loading_field,
                temp_name_f:  mass_loadinf_flux_field
            },
            inplace=True
        )

        return subcatchment_fluxes, concentration_mesh



    def to_dict(self, base_directory: str = None) -> dict:
        """
        This function converts the plume object to a JSON object
        :param base_directory: The base directory for the plume object
        :return: The JSON object
        """

        return {
            "source_strength": self.source_strength,
            "source_location": self.source_location,
            "direction": self.direction,
            "wind_speed": self.wind_speed,
            "turbulent_intensity": self.turbulent_intensity,
            "standard_deviation": self.standard_deviation,
            "plume_type": self.plume_type.name
        }

    @staticmethod
    def from_dict(cls, data: dict, base_directory: str = None) -> 'GaussianPlume':
        """
        This function initializes the plume object from a JSON object
        :param arguments: The JSON object
        :param base_directory: The base directory for the plume object
        :return: GaussianPlume object
        """
        return GaussianPlume(**arguments)