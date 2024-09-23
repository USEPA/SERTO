from typing import List, Tuple
import numpy as np
import unittest
import math
import matplotlib.pyplot as plt
from enum import Enum


class GaussianPlumeOld:

    def __init__(self, source_strength: float, source_location: List[float],
                 direction: float, standard_deviation: List[float]) -> None:
        """
    This function initializes the plume generation object.
    All units are SI units
    :param source_strength: Source strength in kg/s
    :param source_location: An array of the form [x, y]
      specifying the location of the source in meters
    :param direction: Angle in degrees for direction of plume in degrees
      from the x-axis
    :param standard_deviation: An array of the form [downwind, crosswind]
      specifying the extent of the plume
    """

        self.source_strength = source_strength
        self.source_location = source_location
        self.direction = direction
        self.standard_deviation = standard_deviation

        self._direction_vector = np.array([
            math.cos(math.radians(self.direction)),
            math.sin(math.radians(self.direction))
        ])

    def float_compare(self, float1, float2, tolerance=1e-9):
        return abs(float1 - float2) < tolerance

    def overlaps(self, location: List[float]) -> bool:
        """
    This function returns True if the location is inside the plume and False
    otherwise
    :returns: True if the location is inside the plume and False otherwise
    """

        conc = self.concentration(location)
        perc = conc * 100 / self.source_strength
        return perc > 0.01

    def concentration(self, location: List[float]) -> float:
        """
    This function returns the concentration at the location

    """

        downwind, crosswind = self.proj_vectors(location)
        d = np.dot(downwind, self._direction_vector)

        if d >= 0.0:
            downwind_norm = np.linalg.norm(downwind)
            crosswind_norm = np.linalg.norm(crosswind)

            # Using Pasquill (1976) and Irwin (1979)
            # downwind_s = 1 / (1.0 + 0.0308 * downwind_norm ** 0.4548)
            # if downwind_norm <= 10 ** 4
            # else 0.333 * (10000 / downwind_norm) ** 0.5
            # downwind_s = downwind_norm * downwind_s
            # crosswind_s =  1 / (1.0 + 0.0308 * crosswind_norm ** 0.4548)
            # if crosswind_norm <= 10 ** 4
            # else 0.333 * (10000 / crosswind_norm) ** 0.5
            # crosswind_s = crosswind_norm * crosswind_s

            # downwind_s = min(1.0, downwind_norm / (self.standard_deviation[0]))
            crosswind_s = min(1.0, downwind_norm / (self.standard_deviation[0]))

            downwind_s = 1.0
            # crosswind_s = 1.0

            downwind_stdev = downwind_s * self.standard_deviation[0]
            crosswind_stdev = crosswind_s * self.standard_deviation[1]
            if (self.float_compare(location[0], self.source_location[0]) and
                    self.float_compare(location[1], self.source_location[1])):
                conc = self.source_strength
            else:
                conc = self.source_strength * \
                       math.exp(-0.5 * (downwind_norm / downwind_stdev) ** 2) * \
                       math.exp(-0.5 * (crosswind_norm / crosswind_stdev) ** 2)

            return conc
        return 0.0

    def proj_vectors(self, location: List[float]) -> np.array:
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

    def downwind_and_crosswind_distances(
            self, location: List[float]) -> Tuple[float, float]:
        """
    This function returns the downwind and crosswind distances from the source
    :param location: An array of the form [x, y] specifying the location of
    the source
    :return: The downwind and crosswind distances from the source. Downwind
    distance is valid if positive and invalid if <= 0
    """
        r = np.array(location) - np.array(self.source_location)
        d = np.dot(r, self._direction_vector) / np.linalg.norm(
            self._direction_vector)

        c = (np.linalg.norm(r) ** 2 - d ** 2) ** 0.5

        return d, c

    def plot(self, x_start: float, x_end: float, x_divs: int, y_start: float,
             y_end: float, y_divs: int) -> None:
        """
    This function plots the plume

    :param x_start: The starting x coordinate of the plot
    :param x_end: The ending x coordinate of the plot
    :param x_divs: The number of divisions in the x direction
    :param y_start: The starting y coordinate of the plot
    :param y_end: The ending y coordinate of the plot
    :param y_divs: The number of divisions in the y direction

    """
        x_pts = np.linspace(x_start, x_end, x_divs)
        y_pts = np.linspace(y_start, y_end, y_divs)

        xs, ys = np.meshgrid(x_pts, y_pts)
        concs = np.zeros(shape=(x_divs, y_divs), dtype=float)

        for i in range(x_divs):
            for j in range(y_divs):
                concs[i, j] = self.concentration([xs[i, j], ys[i, j]])

        h = plt.contourf(xs, ys, concs)
        plt.colorbar(h)
        plt.show()


class GaussianPlume:

    def __init__(
            self,
            source_strength: float,
            source_location: Tuple[float, float],
            direction: float,
            standard_deviation: Tuple[float, float] = None,
            release_height: float = 1.0e-10,
            wind_speed: float = 10.0,
            lateral_turbulent_intensity: float = 0.25,
            vertical_turbulent_intensity: float = 0.15,
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
    """

        self.source_strength = source_strength
        self.source_location = source_location
        self.direction = direction
        self.release_height = release_height
        self.wind_speed = wind_speed
        self.lateral_turbulent_intensity = lateral_turbulent_intensity
        self.vertical_turbulent_intensity = vertical_turbulent_intensity
        self.aerodynamic_roughness = aerodynamic_roughness

        self.standard_deviation = standard_deviation

        self._direction_vector = np.squeeze(np.array([
            math.cos(math.radians(self.direction)),
            math.sin(math.radians(self.direction))
        ]))

    def overlaps(self, locations: np.array) -> np.array:
        """
    This function returns True if the location is inside the plume and False
    otherwise
    :returns: True if the location is inside the plume and False otherwise
    """

        concs = self.concentration(locations)
        perc = concs * 100 / self.source_strength

        return perc > 0.01

    def concentration(self, locations: np.array) -> np.array:
        """
    This function returns the concentration at the location

    """
        distances = locations[..., :2] - np.array(self.source_location)
        distances = np.linalg.norm(distances, axis=1)

        downwind, crosswind = self.proj_vectors(locations[..., :2])
        d = np.dot(downwind, self._direction_vector)

        concentration = np.zeros(locations.shape[0])
        concentration[d <= 0.0] = 0.0

        downwind_v = downwind[d > 0]
        crosswind_v = crosswind[d > 0]

        downwind_norm = np.linalg.norm(downwind_v, axis=1)
        crosswind_norm = np.linalg.norm(crosswind_v, axis=1)
        crosswind_norm[crosswind_norm < 1] = 1.0

        if self.standard_deviation is None:

            downwind_stdev = self.lateral_turbulent_intensity * downwind_norm
            crosswind_stdev = self.lateral_turbulent_intensity * crosswind_norm

            source_strength = self.source_strength / (
                    2.0 * self.wind_speed * np.pi * np.sqrt(
                downwind_stdev ** 2.0 + crosswind_stdev ** 2.0
            ))

        else:

            crosswind_s = downwind_norm / (self.standard_deviation[0])
            downwind_s = np.full(crosswind_s.shape, 1.0)

            source_strength = self.source_strength
            downwind_stdev = downwind_s * self.standard_deviation[0]
            crosswind_stdev = crosswind_s * self.standard_deviation[1]

        valid_concentrations = (
                source_strength * np.exp(-0.5 * (downwind_norm / downwind_stdev) ** 2) *
                np.exp(-0.5 * (crosswind_norm / crosswind_stdev) ** 2)
        )

        concentration[d > 0] = valid_concentrations
        concentration[distances <= 1.0E-20] = self.source_strength

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

        return a1, a2

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
