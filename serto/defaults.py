# python imports
import os
import json
import os.path
from abc import abstractmethod, ABC
from functools import wraps
from typing import Callable, Any, Dict, Union, Iterable, Tuple

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

    @staticmethod
    def timeseries(
            dataset_arg_name: str = 'dataset',
            processed_dataset_arg_name: str = 'processed_dataset',
            index_column: Union[str, int] = 0,
            parse_dates: Union[str, bool] = True,
    ) -> Callable[..., Any]:
        """
        Decorator that redirects a dataset argument to a pandas timeseries DataFrame.
        :param dataset_arg_name:  The name of the dataset argument in the function signature (default: 'dataset')
        :param processed_dataset_arg_name: The name of the processed dataset argument in the
        function signature (default: 'processed_dataset')
        :param index_column: The index column for the timeseries DataFrame (default: 0)
        :param parse_dates: Whether to parse the dates in the timeseries DataFrame (default: True)
        :return: The decorated function
        """

        def timeseries_dataset_decorator(parent_function: Callable[..., Any]) -> Callable[..., Any]:

            @wraps(parent_function)
            def wrapper(*args, **kwargs):

                if dataset_arg_name not in kwargs:
                    raise ValueError(
                        'The dataset argument is required for this function.'
                        'It can be a pandas DataFrame or a path to a pandas format '
                        'file (e.g., *.csv, *.xlsx, *.parquet, *.json, *.txt, *.h5, etc).'
                    )
                else:
                    dataset = kwargs[dataset_arg_name]
                    if dataset is None:
                        raise ValueError(
                            'The dataset argument is required for this function.'
                            'It can be a pandas DataFrame or a path to a pandas format '
                            'file (e.g., *.csv, *.xlsx, *.parquet, *.json, *.txt, *.h5, etc).'
                        )
                    elif isinstance(dataset, str):
                        if os.path.exists(dataset):
                            # check extension to determine how to read the file
                            if dataset.endswith('.csv'):
                                kwargs[processed_dataset_arg_name] = pd.read_csv(
                                    dataset,
                                    index_col=index_column,
                                    parse_dates=parse_dates,
                                    *args, **kwargs
                                )
                            elif (dataset.endswith('.xlsx') or
                                  dataset.endswith('.xls') or
                                  dataset.endswith('.xlsm') or
                                  dataset.endswith('.xlsb') or
                                  dataset.endswith('.odf') or
                                  dataset.endswith('.ods') or
                                  dataset.endswith('.odt') or
                                  dataset.endswith('.fods') or
                                  dataset.endswith('.fodt') or
                                  dataset.endswith('.xlt') or
                                  dataset.endswith('.xltm') or
                                  dataset.endswith('.xltx')):
                                if 'sheet_name' in kwargs:
                                    kwargs[processed_dataset_arg_name] = pd.read_excel(
                                        dataset,
                                        index_col=index_column,
                                        parse_dates=parse_dates,
                                        sheet_name=kwargs['sheet_name'],
                                        *args, **kwargs
                                    )
                                else:
                                    kwargs[processed_dataset_arg_name] = pd.read_excel(
                                        dataset,
                                        index_col=index_column,
                                        parse_dates=parse_dates,
                                        *args, **kwargs
                                    )
                            elif dataset.endswith('.parquet'):
                                kwargs[processed_dataset_arg_name] = pd.read_parquet(
                                    dataset,
                                    *args, **kwargs
                                )

                            elif dataset.endswith('.json'):
                                kwargs[processed_dataset_arg_name] = pd.read_json(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.txt'):
                                kwargs[processed_dataset_arg_name] = pd.read_csv(
                                    dataset,
                                    index_col=index_column,
                                    parse_dates=parse_dates,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.h5'):
                                if 'hdf5_key' in kwargs:
                                    kwargs[processed_dataset_arg_name] = pd.read_hdf(
                                        dataset,
                                        key=kwargs['hdf5_key'],
                                        *args, **kwargs
                                    )
                                else:
                                    kwargs[processed_dataset_arg_name] = pd.read_hdf(
                                        dataset,
                                        *args, **kwargs
                                    )
                            elif dataset.endswith('.feather'):
                                kwargs[processed_dataset_arg_name] = pd.read_feather(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.dta'):
                                kwargs[processed_dataset_arg_name] = pd.read_stata(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.sas7bdat'):
                                kwargs[processed_dataset_arg_name] = pd.read_sas(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.pkl'):
                                kwargs[processed_dataset_arg_name] = pd.read_pickle(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.html'):
                                kwargs[processed_dataset_arg_name] = pd.read_html(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.sql'):
                                kwargs[processed_dataset_arg_name] = pd.read_sql(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.gbq'):
                                kwargs[processed_dataset_arg_name] = pd.read_gbq(
                                    dataset,
                                    *args, **kwargs
                                )
                            elif dataset.endswith('.xml'):
                                kwargs[processed_dataset_arg_name] = pd.read_xml(
                                    dataset,
                                    *args, **kwargs
                                )

                            else:
                                raise ValueError(
                                    'The dataset argument is required for this function.'
                                    'It can be a pandas DataFrame or a path to a pandas format '
                                    'file (e.g., *.csv, *.xlsx, *.parquet, *.json, *.txt, *.h5, etc).'
                                )

                return parent_function(*args, **kwargs)

            return wrapper

        return timeseries_dataset_decorator

    @staticmethod
    def extra_args(
            config_file_arg_name: str = 'config',
            renamed_args: Dict[str, str] = None,
    ) -> Callable[..., Any]:
        """
        Decorator for the flow analysis module.
        :param config_file_arg_name: The name of the config file argument in the function signature (default: 'config')
        :param renamed_args: A dictionary of renamed arguments to update in the kwargs (default: None)
        :return: The decorated function
        """

        def extra_args_from_config_decorator(
                parent_function: Callable[..., Any],
        ):
            @wraps(parent_function)
            def wrapper(*args, **kwargs):

                if 'config' in kwargs:

                    config_file = kwargs[config_file_arg_name]

                    if config_file is not None and os.path.exists(config_file):
                        if config_file.endswith('.json'):
                            import json
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                        elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                            import yaml
                            with open(config_file, 'r') as f:
                                config = yaml.load(f, Loader=yaml.Loader)
                        else:
                            raise ValueError(f'Unknown file type for config file: {config_file}')

                        if renamed_args is not None:
                            for old_name, new_name in renamed_args.items():
                                if old_name in config:
                                    config[new_name] = config.pop(old_name)

                        kwargs.update(config)
                        kwargs['base_directory'] = os.path.dirname(config_file)

                        del kwargs[config_file_arg_name]

                return parent_function(*args, **kwargs)

            return wrapper

        return extra_args_from_config_decorator

    @staticmethod
    def coordinates(s: str) -> tuple[float, ...]:
        """
        Convert a string of coordinates to a tuple of floats
        :param s: The string of coordinates
        :return: The tuple of floats
        """
        return tuple(map(float, s.split(',')))

    @staticmethod
    def named_coordinates(s: str) -> Tuple[str, float, float]:
        """
        Convert a string of named coordinates to a tuple of strings and floats
        :param s: The string of named coordinates
        :return: The tuple of strings and floats
        """
        name, x, y = s.split(',')
        return name, float(x), float(y)


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
    def to_dict(self, base_directory: str = None, *args, **kwargs) -> dict:
        """
        Convert the object to a dictionary
        :param base_directory: The base directory for relative paths
        :return: The dictionary
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[Any, Any], base_directory: str = None, *args, **kwargs) -> 'IDictable':
        """
        Create an object from a dictionary representation
        :param data: The dictionary
        :param base_directory: The base directory for relative paths
        :return: The object
        """
        raise NotImplementedError
