"""
sero.analysis.plumes.PlumeEventMatrix class for generating plume event matrices
"""
# python imports
from typing import Tuple, List, Any, Union, Dict
from enum import Enum
from datetime import datetime
import math
import itertools

# third-party imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import NDArray

# local imports
from ...swmm import SpatialSWMM
from . import GaussianPlume


class PlumeEventMatrix:
    """
    This class generates a plume event matrix
    TODO: Add wind event probabilities to the plume event matrix
    """

    def __init__(
            self,
            swmm_input_file: str,
            crs: str,
            contaminant_loading_spatial_resolution: float,
            release_time: datetime,
            contaminant_name: str,
            contaminant_units: str,
            plume_type: GaussianPlume.PlumeType,
            contaminant_amounts: Union[float, Tuple[float, ...], List[float]],
            wind_directions: Union[float, Tuple[float, ...], List[float]],
            wind_speeds: Union[float, Tuple[float, ...], List[float]] = None,
            standard_deviations: List[Tuple[float, float]] = None,
            turbulent_intensities: Union[float, Tuple[float, ...], List[float]] = None,
            release_location_element_types: List[str] = ('JUNCTIONS',),
            release_locations: Dict[str, Tuple[float, float]] = None,
            release_grid: Tuple[float, float] = None,
    ) -> None:
        """
        Constructor for the plume event matrix object
        :param swmm_input_file:
        :param contaminant_loading_spatial_resolution:
        :param release_time:
        :param contaminant_name:
        """
        self.swmm_input_file = swmm_input_file
        self.contaminant_loading_spatial_resolution = contaminant_loading_spatial_resolution
        self.release_time = release_time
        self.contaminant_name = contaminant_name
        self.contaminant_units = contaminant_units
        self.plume_type = plume_type

        if isinstance(contaminant_amounts, float) or isinstance(contaminant_amounts, int):
            contaminant_amounts = [float(contaminant_amounts)]

        self.contaminant_amounts = contaminant_amounts

        if isinstance(wind_directions, float) or isinstance(wind_directions, int):
            wind_directions = [float(wind_directions)]

        self.wind_directions = wind_directions

        if isinstance(wind_speeds, float) or isinstance(wind_speeds, int):
            wind_speeds = [float(wind_speeds)]

        self.wind_speeds = wind_speeds

        self.standard_deviations = standard_deviations

        if isinstance(turbulent_intensities, float) or isinstance(turbulent_intensities, int):
            turbulent_intensities = [float(turbulent_intensities)]

        self.turbulent_intensities = turbulent_intensities

        self.release_location_element_types = release_location_element_types
        self.release_locations = release_locations
        self.release_grid = release_grid
        self.crs = crs

        self.swmm_model = SpatialSWMM.read_model(model_path=self.swmm_input_file, crs=self.crs)
        self._model_bounds = np.array([
            self.swmm_model.nodes.total_bounds,
            self.swmm_model.links.total_bounds,
            self.swmm_model.subcatchments.total_bounds
        ])

        self._model_bounds = {
            'min_x': self._model_bounds[:, 0].min(),
            'min_y': self._model_bounds[:, 1].max(),
            'max_x': self._model_bounds[:, 2].min(),
            'max_y': self._model_bounds[:, 3].max()
        }

        x_range = self._model_bounds['max_x'] - self._model_bounds['min_x']
        y_range = self._model_bounds['max_y'] - self._model_bounds['min_y']

        if (
                (contaminant_loading_spatial_resolution > x_range) or
                (contaminant_loading_spatial_resolution > y_range)
        ):
            raise ValueError("Contaminant loading spatial resolution is too large")

        self._release_locations: List = []
        self._validate_release_locations()
        self._plume_event_summaries = self.generate_plume_event_summaries()

    def _validate_release_locations(self):

        if (
                (self.release_location_element_types is None or len(self.release_location_element_types) == 0) and
                (self.release_locations is None or len(self.release_locations) == 0) and
                self.release_grid is None
        ):
            raise ValueError("Release location element types must be provided")

        for r in self.release_location_element_types:
            if r in ('JUNCTIONS', 'OUTFALLS', 'STORAGE', 'DIVIDERS'):
                junctions = self.swmm_model.nodes[self.swmm_model.nodes['NodeType'] == r]
                for index, row in junctions.iterrows():
                    release_data = {
                        'ReleaseType': 'SWMM_NODE',
                        'ReleaseElementType': r,
                        'ReleaseID': index,
                        'ReleaseLocation X': row.geometry.x,
                        'ReleaseLocation Y': row.geometry.y,

                    }
                    self._release_locations.append(release_data)

            elif r == 'SUBCATCHMENTS':
                subcatchments = self.swmm_model.subcatchments
                for index, row in subcatchments.iterrows():
                    release_data = {
                        'ReleaseType': 'SWMM_SUBCATCHMENT',
                        'ReleaseElementType': r,
                        'ReleaseID': index,
                        'ReleaseLocation X': row.geometry.centroid.x,
                        'ReleaseLocation Y': row.geometry.centroid.y
                    }
                    self._release_locations.append(release_data)

        if self.release_locations is not None:
            for k, v in self.release_locations.items():
                release_data = {
                    'ReleaseType': 'USER_DEFINED',
                    'ReleaseElementType': 'USER_DEFINED',
                    'ReleaseID': k,
                    'ReleaseLocation X': v[0],
                    'ReleaseLocation Y': v[1]
                }
                self._release_locations.append(release_data)

        if self.release_grid is not None:

            x_grid = np.arange(
                self._model_bounds['min_x'] + self.release_grid[0] / 2.0,
                self._model_bounds['max_x'],
                self.release_grid[0]
            )

            y_grid = np.arange(
                self._model_bounds['min_y'] + self.release_grid[1] / 2.0,
                self._model_bounds['max_y'],
                self.release_grid[1]
            )

            for i, x in enumerate(x_grid):
                for j, y in enumerate(y_grid):
                    release_data = {
                        'ReleaseType': 'GRID',
                        'ReleaseElementType': 'GRID',
                        'ReleaseID': f'{i}_{j}',
                        'ReleaseLocation X': x,
                        'ReleaseLocation Y': y
                    }
                    self._release_locations.append(release_data)

    @property
    def plume_events_summary(self) -> pd.DataFrame:
        """
        This function generates plume event summaries
        :return: A pandas dataframe with the plume event summaries
        """
        return self._plume_event_summaries

    @property
    def model_bounds(self) -> Dict:
        """
        This function returns the model bounds
        :return: A dictionary with the model bounds
        """
        return self._model_bounds

    def get_plume_event(self, index: int) -> GaussianPlume:
        """
        :param index:
        :return:
        """
        row = self._plume_event_summaries.iloc[index]

        plume = GaussianPlume(
            source_strength=row['ContaminantAmount'],
            source_location=(row['ReleaseLocation X'], row['ReleaseLocation Y']),
            direction=row['WindDirection'],
            standard_deviation=(
                row['StandardDeviation Downwind'],
                row['StandardDeviation Crosswind']
            ),
            plume_type=self.plume_type
        )

        return plume

    def generate_plume_event_summaries(self) -> pd.DataFrame:
        """
        This function generates plume event summaries
        :return: A pandas dataframe with the plume event summaries
        :TODO Add wind event probabilities
        """
        plume_summaries = []

        if not self.standard_deviations is None:
            for plumes in itertools.product(
                    self.contaminant_amounts,
                    self.wind_directions,
                    self.standard_deviations,
                    self._release_locations
            ):
                row = {
                    'ContaminantAmount': plumes[0],
                    'WindDirection': plumes[1],
                    'StandardDeviation Downwind': plumes[2][0],
                    'StandardDeviation Crosswind': plumes[2][1],
                    'PlumeType': self.plume_type.name,
                    'ReleaseType': plumes[3]['ReleaseType'],
                    'ReleaseElementType': plumes[3]['ReleaseElementType'],
                    'ReleaseID': plumes[3]['ReleaseID'],
                    'ReleaseLocation X': plumes[3]['ReleaseLocation X'],
                    'ReleaseLocation Y': plumes[3]['ReleaseLocation Y'],
                }

                plume_summaries.append(row)

        return pd.DataFrame(plume_summaries)

    def to_dict(self):
        output = {}
        output['swmm_input_file'] = self.swmm_input_file
        output['contaminant_loading_resolution'] = {
            'x_resolution': self.contaminant_loading_resolution[0],
            'y_resolution': self.contaminant_loading_resolution[1]
        }

    def from_dict(self, data: Dict):
        pass
