# python imports
import json
import os.path
from abc import abstractmethod, ABC
from typing import Callable, Any, Dict

# third-party imports
import numpy as np
import pandas as pd


# project imports

class SERTODefaults:
    """
    Default constants for the SERTO package
    """
    # Default number of threads to use for each SWMM simulation
    THREADS_PER_SIMULATION: int = 4

    # Precipitation frequency data server url
    NOAA_PFDS_REST_SERVER_URL: str = r'https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/'

    @classmethod
    def timeseries_dataset_decorator(cls, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator for the flow analysis module.
        :param func: The function to decorate
        :return: The decorated function
        """

        def wrapper(*args, **kwargs):

            if 'dataset' not in kwargs:
                raise ValueError('The dataset argument is required for this function')
            else:
                dataset = kwargs['dataset']

                if dataset is None:
                    raise ValueError('The dataset argument is required for this function')
                elif isinstance(dataset, str):
                    if os.path.exists(dataset):
                        kwargs['dataset'] = pd.read_csv(dataset, index_col=0, parse_dates=True)

            return func(*args, **kwargs)

        return wrapper

    @classmethod
    def dataset_decorator(cls, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator for the flow analysis module.
        :param func: The function to decorate
        :return: The decorated function
        """

        def wrapper(*args, **kwargs):

            if 'dataset' not in kwargs:
                raise ValueError('The dataset argument is required for this function')
            else:
                dataset = kwargs['dataset']

                if dataset is None:
                    raise ValueError('The dataset argument is required for this function')
                elif isinstance(dataset, str):
                    if os.path.exists(dataset):
                        kwargs['dataset'] = pd.read_csv(dataset, index_col=0)

            return func(*args, **kwargs)

        return wrapper


class SERTONumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class IDictable(ABC):
    """
    Interface for objects that can be converted to and from a dictionary
    """

    @abstractmethod
    def to_dict(self, base_directory: str = None) -> dict:
        """
        Convert the object to a dictionary
        :param base_directory: The base directory for relative paths
        :return: The dictionary
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[Any, Any], base_directory: str = None) -> 'IDictable':
        """
        Create an object from a dictionary representation
        :param data: The dictionary
        :param base_directory: The base directory for relative paths
        :return: The object
        """
        raise NotImplementedError
