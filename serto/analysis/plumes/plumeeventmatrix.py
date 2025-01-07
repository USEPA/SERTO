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
from . import GaussianPlume

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
                NDArray[Tuple[int, int], np.float64],
                List[Tuple[float, float]],
            ] = None,
            turbulent_intensities: Union[NDArray[np.float64], Tuple[float, ...], List[float]] = None,
            aerodynamic_roughness: Union[NDArray[np.float64], Tuple[float, ...], List[float]] = None,
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
        self.aerodynamic_roughness = aerodynamic_roughness
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