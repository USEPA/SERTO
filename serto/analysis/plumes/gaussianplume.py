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

# local imports
from ...swmm import SpatialSWMM

class GaussianPlume:
    class PlumeType(Enum):
        EMPIRICAL = 1
        PHYSICS_BASED = 2

    def __init__(
            self,
            source_strength: float,
            source_location: Tuple[float, float],
            direction: float,
            standard_deviation: Tuple[float, float] = None,
            wind_speed: float = 10.0,
            turbulent_intensity: float = 0.25,
            aerodynamic_roughness: float = 0.25,
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
    :param direction: Angle in degrees for direction of plume in degrees
      from the x-axis
    :param standard_deviation: An array of the form [downwind, crosswind]
      specifying the extent of the plume
    :param wind_speed : Wind speed in m/s
    :param turbulent_intensity : Lateral turbulent intensity in m/s
    """

        self.source_strength = source_strength
        self.source_location = source_location
        self.direction = direction
        self.wind_speed = wind_speed
        self.turbulent_intensity = turbulent_intensity
        self.aerodynamic_roughness = aerodynamic_roughness

        self.standard_deviation = standard_deviation

        self._direction_vector = np.squeeze(np.array([
            math.cos(math.radians(self.direction)),
            math.sin(math.radians(self.direction))
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
    :returns: True if the location is inside the plume and False otherwise
    """

        concs = self.concentration(locations)
        perc = concs * 100 / self.source_strength

        return perc > 0.001

    def concentration(self, locations: np.array) -> np.array:
        """
    This function returns the concentration at the location

    """
        distances = locations[..., :2] - np.array(self.source_location)
        distances = np.linalg.norm(distances, axis=1)

        downwind, crosswind, radis = self.proj_vectors(locations[..., :2])
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

    def proj_vectors(self, locations: np.array) -> np.array:
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

    def proj_vectors_old(self, location: Tuple[float, float]) -> np.array:
        """
        This function returns the distance from the source
        :param location: An array of the form [x, y]
          specifying the location of the source
        :return: The distance from the source. Distance is valid
          if positive and invalid if <= 0
        """
        r = np.array(location) - np.array(self.source_location)
        a1 = np.dot(r, self._direction_vector) * self._direction_vector
        a2 = r - a1

        return a1, a2

    def plot(self, x_pts: np.array, y_pts: np.array, concentrations: np.array) -> None:
        """
        This function plots the plume

        """
        h = plt.contourf(x_pts, y_pts, concentrations)
        plt.colorbar(h)
        plt.show()

class PlumeEventMatrix:

    def __init__(
            self,
            swmm_input_file: str,
            crs: str,
            contaminant_loading_resolution: Tuple[float, float],
            release_time: datetime,
            contaminant_name: str,
            contaminant_units: str,
            plume_type: GaussianPlume.PlumeType,
            contaminant_amounts: Union[NDArray[np.float64], Tuple[float, ...], List[float]],
            wind_directions: Union[NDArray[np.float64], Tuple[float, ...], List[float]],
            wind_speeds: Union[NDArray[np.float64], Tuple[float, ...], List[float]],
            standard_deviations: Union[
                NDArray[np.float64],
                List[Tuple[float, float]],
            ] = None,
            turbulent_intensities: Union[NDArray[np.float64], Tuple[float, ...], List[float]] = None,
            aerodynamic_roughnesses: Union[NDArray[np.float64], Tuple[float, ...], List[float]] = None,
            release_location_element_types: List[str] = ('junctions', 'storages',),
            release_locations: Dict[str, Tuple[float, float]] = None,
            release_grid: Tuple[float, float] = None,
    ) -> None:
        """
        Constructor for the plume event matrix object
        :param swmm_input_file:
        :param contaminant_loading_resolution:
        :param release_time:
        :param contaminant_name:
        """
        self.swmm_input_file = swmm_input_file
        self.contaminant_loading_resolution = contaminant_loading_resolution
        self.release_time = release_time
        self.contaminant_name = contaminant_name
        self.contaminant_units = contaminant_units
        self.plume_type = plume_type
        self.contaminant_amounts = contaminant_amounts
        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds
        self.standard_deviations = standard_deviations
        self.turbulent_intensities = turbulent_intensities
        self.aerodynamic_roughnesses = aerodynamic_roughnesses
        self.release_location_element_types = release_location_element_types
        self.release_locations = release_locations
        self.release_grid = release_grid
        self.crs = crs

        self.swmm_model = SpatialSWMM.read_model(model_path=self.swmm_input_file, crs=self.crs)


    def num_scenarios(self):
        return (
            len(self.contaminant_amounts) *
            len(self.wind_directions) *
            len(self.wind_speeds) *
            len(self.standard_deviations) * len(self.turbulent_intensities) * len(self.aerodynamic_roughnesses)
        )
    def to_dict(self):
        output = {}
        output['swmm_input_file'] = self.swmm_input_file
        output['contaminant_loading_resolution'] = {
            'x_resolution': self.contaminant_loading_resolution[0],
            'y_resolution': self.contaminant_loading_resolution[1]
        }

    def from_dict(self, data: Dict):
        pass