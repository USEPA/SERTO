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
import matplotlib.pyplot as plt
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
        EMPIRICAL = 1
        PHYSICS_BASED = 2

    def __init__(
            self,
            source_strength: float,
            source_location: Tuple[float, float],
            wind_direction: float,
            standard_deviation: Tuple[float, float] = None,
            wind_speed: float = 10.0,
            turbulent_intensity: float = 0.25,
            *args,
            **kwargs

    ) -> None:
        """
        This function initializes the plume generation object.
        All units are SI units. If standard deviation is not provided the
        physics-based formulation is used. If standard deviation is provided
        the empirical formulation is used.

        :param source_strength: Source strength in kg/s
        :param source_location: An array of the form [x, y]
          specifying the location of the source in meters
        :param wind_direction: Angle in degrees for direction of plume in degrees
          from the x-axis
        :param standard_deviation: An array of the form [downwind, crosswind]
          specifying the extent of the plume
        :param wind_speed : Wind speed in m/s
        :param turbulent_intensity : Lateral turbulent intensity in m/s
        """

        self.source_strength = source_strength
        self.source_location = source_location
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.turbulent_intensity = turbulent_intensity

        self.standard_deviation = standard_deviation

        self._direction_vector = np.squeeze(np.array([
            math.cos(math.radians(self.wind_direction)),
            math.sin(math.radians(self.wind_direction))
        ]))

        self._plume_type = self.PlumeType.PHYSICS_BASED if self.standard_deviation is None else self.PlumeType.EMPIRICAL

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
        concentration[d < 0.0] = 0.0

        downwind_v = downwind[d >= 0]
        crosswind_v = crosswind[d >= 0]

        downwind_norm = np.linalg.norm(downwind_v, axis=1)
        crosswind_norm = np.linalg.norm(crosswind_v, axis=1)

        if self.plume_type == self.PlumeType.PHYSICS_BASED:
            # physics-based formulation

            downwind_standard_dev = self.turbulent_intensity * downwind_norm
            crosswind_standard_dev = self.turbulent_intensity * crosswind_norm

            crosswind_weight = downwind_norm / downwind_standard_dev
            crosswind_standard_dev *= crosswind_weight

            source_strength = self.source_strength / (
                    2.0 * np.pi * self.wind_speed * downwind_standard_dev * crosswind_standard_dev
            )

        else:
            downstream_weight = 1.0
            crosswind_weight = downwind_norm / self.standard_deviation[0]

            downwind_standard_dev = self.standard_deviation[0] * downstream_weight
            crosswind_standard_dev = self.standard_deviation[1] * crosswind_weight

            source_strength = self.source_strength

        valid_concentrations = source_strength * np.exp(
            -0.5 * (crosswind_norm / crosswind_standard_dev) ** 2.0
        ) * np.exp(
            -0.5 * (downwind_norm / downwind_standard_dev) ** 2.0
        )

        concentration[d >= 0] = valid_concentrations
        # concentration[distances <= 1.0E-20] = self.source_strength

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

    def concentrations_polygon(
            self,
            x_min:float ,
            x_max:float ,
            y_min:float ,
            y_max:float ,
            resolution:float,
            crs: str,
            concentration_field_name: str = "Concentration"
    ) -> gpd.GeoDataFrame:
        """
        This function returns the concentrations for each point in a square polygon

        :param x_min: The minimum x location of the domain
        :param x_max: The maximum x location of the domain
        :param y_min: The minimum y location of the domain
        :param y_max: The maximum y location of the domain
        :param resolution: The resolution of the grid
        :param crs: The coordinate reference system of the grid
        :param concentration_field_name: The name of the concentration field in the GeoDataFrame
        :return: A GeoDataFrame with the concentration field
        """

        half_resolution = resolution / 2.0
        x_pts = np.arange(x_min + half_resolution, x_max, resolution)
        y_pts = np.arange(y_min + half_resolution, y_max, resolution)

        x_pts, y_pts = np.meshgrid(x_pts, y_pts)

        locations = np.array([x_pts, y_pts]).T.reshape(-1, 2)
        concentrations = self.concentration(locations)

        concentration_polygons = gpd.GeoDataFrame(
            {
                "geometry":  [
                    Polygon([
                        (x - half_resolution , y - half_resolution),
                        (x + half_resolution, y - half_resolution),
                        (x + half_resolution, y + half_resolution),
                        (x - half_resolution, y + half_resolution)
                    ])
                    for x, y in locations
                ],
                "concentration": concentrations.flatten(),
            },
            crs=crs
        )

        # concentration_polygons.set_crs(crs, inplace=True)

        return concentration_polygons

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
    def from_dict(cls, data: dict, base_directory: str = None) -> GaussianPlume:
        """
        This function initializes the plume object from a JSON object
        :param arguments: The JSON object
        :param base_directory: The base directory for the plume object
        :return: GaussianPlume object
        """
        return GaussianPlume(**arguments)