import copy
import os
from typing import List, Tuple, Iterable, Dict, Any
import shutil
import logging
from pathlib import Path

import yaml
import pandas as pd
import re

from .. analysis import GaussianPlume
from .. import utils
from .. import DEFAULT_THREADS_PER_SIMULATION


class InputFileGenerator:

    def __init__(self):
        pass

    @staticmethod
    def replace_options_section(
            dataframe: pd.DataFrame,
            input_file_lines: List[str],
            section_start_index: int = 0
    ) -> Tuple[List[str], int]:
        """
        Replace the [OPTIONS] section in the input file with the content of the DataFrame.
        :param dataframe: DataFrame containing the data to replace the [OPTIONS] section.
        :param input_file_lines: The content of the input file as a list of strings.
        :param section_start_index: The index of the line where the [OPTIONS] section starts.
        :return: Tuple of the modified content of the input file and the index of the line after the [OPTIONS] section.
        """

        start_tag = '[OPTIONS]'
        tag_found = False

        for i in range(len(input_file_lines)):
            line = input_file_lines[i]
            stripped_line = line.strip()
            if stripped_line == start_tag:
                section_start_index = i
                tag_found = True
            elif tag_found and stripped_line.startswith(';;'):
                pass
            elif tag_found and stripped_line.startswith('['):
                break
            elif tag_found and not (
                    stripped_line.startswith(';') or stripped_line == "" or stripped_line.startswith('[')):
                field_name, field_value = re.split("\t|\s+", stripped_line)
                if dataframe['Name'].str.contains(field_name).any():
                    replace_line = f"{field_name}\t{dataframe.loc[dataframe['Name'] == field_name, 'Value'].values[0]}\n"
                    input_file_lines[i] = replace_line

        return input_file_lines, section_start_index

    @staticmethod
    def add_time_series_file(
            input_file_lines: List[str],
            timeseries_line: str,
            section_start_index: int = 0
    ) -> Tuple[List[str], int]:
        """
        Replace the raingage section in the input file with the content of the hot start file.
        :param input_file_lines: The content of the input file as a list of strings.
        :param timeseries_line: The line to insert into the TIMESERIES section.
        :param section_start_index: The index of the line where the TIMESERIES section starts.
        :return: Tuple of the modified input file lines and the index of the line after the TIMESERIES section.
        """

        start_tag = '[TIMESERIES]'
        tag_found = False

        i = section_start_index
        while section_start_index <= i <= len(input_file_lines):
            line = input_file_lines[i]
            stripped_line = line.strip()
            if stripped_line == start_tag:
                tag_found = True
                section_start_index = i
            elif tag_found and stripped_line.startswith(';;'):
                pass
            elif tag_found and stripped_line.startswith('['):
                break
            elif tag_found and not (
                    stripped_line.startswith(';') or stripped_line == "" or stripped_line.startswith('[')):
                input_file_lines.insert(i, timeseries_line + '\n')
                break

            i += 1

        return input_file_lines, section_start_index

    @staticmethod
    def replace_loadings_section(
            loadings: pd.DataFrame,
            input_file_lines: List[str],
            section_start_index=0
    ) -> Tuple[List[str], int]:
        """
        Replace the [LOADINGS] section in the input file with the content of the DataFrame.
        :param loadings: DataFrame containing the data to replace the [LOADINGS] section.
        :param input_file_lines: The content of the input file as a list of strings.
        :param section_start_index: The index of the line where the [LOADINGS] section starts.
        :return: Tuple of the modified content of the input file and the index of the line after the [LOADINGS] section.
        """
        start_tag = '[LOADINGS]'

        tag_found = False
        found_insertion_point = False
        insertion_point = 0

        i = section_start_index

        while section_start_index <= i <= len(input_file_lines):
            line = input_file_lines[i]
            stripped_line = line.strip()
            if stripped_line == start_tag:
                tag_found = True
                section_start_index = i
            elif tag_found and stripped_line.startswith(';;'):
                pass
            elif tag_found and stripped_line.startswith('['):
                for index, row in loadings.iterrows():
                    insert_line = f"{row['NAME']}\t{row['Pollutant']}\t{row['Buildup']}\n"

                    input_file_lines.insert(
                        insertion_point,
                        insert_line
                    )
                    insertion_point += 1
                break
            elif tag_found and not (
                    stripped_line.startswith(';') or stripped_line == "" or stripped_line.startswith('[')):
                if not found_insertion_point:
                    found_insertion_point = True
                    insertion_point = i
                del input_file_lines[i]
                i = i - 1

            i += 1

        return input_file_lines, section_start_index

    @staticmethod
    def replace_hot_start_section(
            input_file_lines: List[str],
            hots_start_file_content: str,
            section_start_index: int = 0
    ) -> Tuple[List[str], int]:
        """
        Replace the hot start section in the input file with the content of the hot start file.
        :param section_start_index:
        :param input_file_lines:
        :param hots_start_file_content:
        :return: Tuple of the modified input file lines and the index of the line after the hot start section.
        """

        start_tag = '[FILES]'
        tag_found = False

        first_line_index = 0
        first_line = False

        i = section_start_index
        while section_start_index <= i <= len(input_file_lines):
            line = input_file_lines[i]
            stripped_line = line.strip()
            if stripped_line == start_tag:
                tag_found = True
                section_start_index = i
            elif tag_found and stripped_line.startswith(';;'):
                pass
            elif tag_found and stripped_line.startswith('['):
                input_file_lines.insert(first_line_index, hots_start_file_content + '\n')
                break
            elif tag_found and not (
                    stripped_line.startswith(';') or stripped_line == "" or stripped_line.startswith('[')):
                if not first_line:
                    first_line = True
                    first_line_index = i
                del input_file_lines[i]
                i = i - 1

            i += 1

        return input_file_lines, section_start_index

    @staticmethod
    def read_yaml_file(file_path):
        """ Function to read YAML data from a file

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
        with open(file_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        return yaml_data


    @staticmethod
    def generate_scenario_matrix(
            rainfall_events: pd.DataFrame,
            wind_data: pd.DataFrame,
            wind_sample_size: int,
            release_points: Iterable[Tuple[float, float]],
            release_concentrations: Iterable[float],
            release_heights: Iterable[float],
    ) -> pd.DataFrame:
        """
        Generate a scenario matrix for the ensemble.
        :param rainfall_events: A pandas dataframe containing the rainfall events. Columns should include
        'start' and 'end' fields to the start and end times of the events and a 'hotstart' field to the
        hot start file name to use for the event.
        :param wind_data:  A pandas dataframe containing the wind data. Columns should include 'speed' and
        'direction' fields. Direction is 0 degrees for North, 90 degrees for East, 180 degrees for South,
        and 270 degrees for West. Speed is in meters per second.
        :param wind_sample_size: The number of wind samples to generate for each wind direction.
        :param release_points: An iterable of tuples containing the x and y coordinates of the release points.
        :param release_concentrations: An iterable of the concentrations of the release points.
        :param release_heights: An iterable of the heights of the release points.
        :return: A pandas dataframe containing the scenario matrix.
        """
        pass

    @staticmethod
    def generate_release_matrix(
            release_points: pd.DataFrame,
            wind_directions: Iterable[float],
            release_concentrations: Iterable[float],
    ) -> pd.DataFrame:
        """
        Generate a release matrix for the ensemble.
        :param release_points: A pandas dataframe containing the release points. Columns should include
        'NAME',  'X' and 'Y' fields to the x and y coordinates of the release points.
        :param wind_directions:  An iterable of wind directions in degrees.
        :param release_concentrations: An iterable of the concentrations of the release points.
        :return: A pandas dataframe containing the release matrix.
        """

        release_matrix = pd.DataFrame(columns=[
            'Release Name', 'Release X', 'Release Y', 'Release Concentration',
            'Wind_Direction', 'Name', 'X', 'Y', 'Concentration'
        ])

        for index, row in release_points.iterrows():
            pass

        return release_matrix

    @staticmethod
    def create_input_files(
            config: Dict[Any, Any],
            config_directory: str,
            swmm_io_path: str,
    ) -> pd.DataFrame:
        """
        Configure SWMM Runs for the various scenario configurations to be evaluated in the ensemble.
        :param config: The configuration data for the ensemble.
        :param config_directory:
        :param swmm_io_path:
        """

        # Load the YAML data from the file
        rainfall_hot_start_file_names = config.get('Rainfall_Hotstart_filenames')
        rg_name = config.get('raingage_name')

        #   # Access and store values in appropriate variables
        standard_deviations = config.get("standard_deviations")

        wind_direction_degree = config.get("wind_direction_degree")
        contaminant_size_in_pico_curie = config.get("contaminant_size_in_picocurie", 0)
        base_inp_file = config.get("inp_file")

        paths = config.get("paths")

        input_paths = utils.return_path(config_directory, paths['input'])
        network_name = config.get("model_name")
        original_inp_file = os.path.join(input_paths, base_inp_file)
        sub_catchments_centers_file = config.get("subcatchments_centers_file", "")

        csv_file_path = os.path.join(input_paths, sub_catchments_centers_file)
        temp_df = pd.read_csv(csv_file_path)

        report_time_in_minutes = int(config.get("reporting_time_in_min", 5))

        # Start Time and End Time from rainfall meta file
        rainfall_meta_file_name = config.get("rainfall_metadata_file")
        rainfall_meta_file_w_path = os.path.join(input_paths,
                                                 rainfall_meta_file_name)
        rainfall_meta_data = pd.read_csv(rainfall_meta_file_w_path, parse_dates=['start', 'end'])

        threads_per_simulation = config.get("num_threads_per_simulation", DEFAULT_THREADS_PER_SIMULATION)

        # Rainfall Path + Rainfall File name + .dat
        rainfall_hot_start_paths = paths['rainfall_hotstart_input']
        rg_file_type = ".dat"
        hs_file_type = ".hsf"

        input_file_lines = []
        with open(original_inp_file, 'r') as file:
            input_file_lines = file.readlines()

        hot_start_section_start_index = 0
        rain_gage_section_start_index = 0
        loading_section_index = 0
        options_section_index = 0
        scenario_index = 0
        scenarios = []

        for rainfall_name in rainfall_hot_start_file_names:

            rainfall_hot_start_input_file_lines = input_file_lines.copy()

            hot_start_file = f"{rainfall_name}{hs_file_type}"
            hot_start_w_path = utils.return_path(
                config_directory,
                os.path.join(rainfall_hot_start_paths, hot_start_file)
            )
            hot_start_line = f'USE HOTSTART "{hot_start_w_path}\n'
            rainfall_hot_start_input_file_lines, hot_start_section_start_index = InputFileGenerator.replace_hot_start_section(
                rainfall_hot_start_input_file_lines, hot_start_line, hot_start_section_start_index
            )

            rain_gage_file = f"{rainfall_name}{rg_file_type}"

            rainfall_file_w_path = utils.return_path(
                config_directory,
                os.path.join(rainfall_hot_start_paths, rain_gage_file)
            )
            rain_gage_line = f'{rg_name}\tFILE "{rainfall_file_w_path}'

            rainfall_hot_start_input_file_lines, rain_gage_section_start_index = InputFileGenerator.add_time_series_file(
                rainfall_hot_start_input_file_lines, rain_gage_line, rain_gage_section_start_index
            )

            # Get the start date and time and the end date and time based on the meta
            # data table and split the date and times for each.
            selected_row = rainfall_meta_data[rainfall_meta_data['rainfall'] == rainfall_name]
            start_date_time = selected_row.iloc[0]['start']
            end_date_time = selected_row.iloc[0]['end']

            start_rainfall_event_date = start_date_time.strftime('%m/%d/%Y')
            start_rainfall_event_time = start_date_time.strftime('%H:%M')

            end_rainfall_event_date = end_date_time.strftime('%m/%d/%Y')
            end_rainfall_event_time = end_date_time.strftime('%H:%M')

            options_dataframe = pd.DataFrame(
                columns=['Name', 'Value'],
                data=[
                    ['START_DATE', start_rainfall_event_date],
                    ['START_TIME', start_rainfall_event_time],
                    ['REPORT_START_DATE', start_rainfall_event_date],
                    ['REPORT_START_TIME', start_rainfall_event_time],
                    ['END_DATE', end_rainfall_event_date],
                    ['END_TIME', end_rainfall_event_time],
                    ['THREADS', threads_per_simulation],
                    ['REPORT_STEP', f"00:{report_time_in_minutes:02d}"],
                ]
            )

            for wind_direction in wind_direction_degree:

                for index, row in temp_df.iterrows():

                    name = row['NAME']
                    x_value = row['X']
                    y_value = row['Y']

                    rain_gage_name = rain_gage_file.split(".")

                    for standard_deviation in standard_deviations:

                        data_df = {
                            'NAME': name,
                            'X': x_value,
                            'Y': y_value,
                        }

                        scenario_file_lines = rainfall_hot_start_input_file_lines.copy()

                        sdx = standard_deviation['x']
                        sdy = standard_deviation['y']

                        data_df['Standard_Deviation_X'] = sdx
                        data_df['Standard_Deviation_Y'] = sdy
                        data_df['Detonation_Location'] = name
                        data_df['Detonation_Location_X'] = x_value
                        data_df['Detonation_Location_Y'] = y_value
                        data_df['Rainfall_Event'] = rainfall_name
                        data_df['Wind_Direction'] = wind_direction
                        scenario_index += 1

                        new_inp_file = os.path.join(
                            swmm_io_path,
                            f'{network_name}_{name}_{rain_gage_name[0]}_wind_direction_{wind_direction}_sd_{sdx}_{sdy}.inp'
                        )

                        data_df['SWMM_InputFiles'] = new_inp_file
                        data_df['Scenario'] = Path(new_inp_file).stem

                        plume = GaussianPlume(
                            source_strength=contaminant_size_in_pico_curie,
                            source_location=(x_value, y_value),
                            direction=wind_direction,
                            standard_deviation=(sdx, sdy)
                        )

                        concentration_values = plume.concentration(locations=temp_df[['X', 'Y']].values)

                        temp_df['Pollutant'] = "Cesium"
                        temp_df['Buildup'] = concentration_values

                        scenario_file_lines, loading_section_index = InputFileGenerator.replace_loadings_section(
                            temp_df,
                            scenario_file_lines,
                            loading_section_index
                        )

                        scenario_file_lines, options_section_index = InputFileGenerator.replace_options_section(
                            options_dataframe,
                            scenario_file_lines,
                            options_section_index
                        )

                        with open(new_inp_file, 'w') as file:
                            file.writelines(scenario_file_lines)

                        scenarios.append(data_df)

        scenarios_table = pd.DataFrame(scenarios)

        return scenarios_table

    @staticmethod
    def create_input_files_from_table(
            config,
            config_directory: str,
            scenarios_table: pd.DataFrame,
            swmm_io_path: str
    ) -> pd.DataFrame:

        # Load the YAML data from the file
        # rainfall_hot_start_file_names = config.get('Rainfall_Hotstart_filenames')
        rg_name = config.get('raingage_name')

        #   # Access and store values in appropriate variables
        # standard_deviations = config.get("standard_deviations")

        # wind_direction_degree = config.get("wind_direction_degree")
        contaminant_size_in_pico_curie = config.get("contaminant_size_in_picocurie", 0)
        base_inp_file = config.get("inp_file")

        paths = config.get("paths")

        input_paths = utils.return_path(config_directory, paths['input'])
        network_name = config.get("model_name")
        original_inp_file = os.path.join(input_paths, base_inp_file)
        # sub_catchments_centers_file = config.get("subcatchments_centers_file", "")

        # csv_file_path = os.path.join(input_paths, sub_catchments_centers_file)
        # temp_df = pd.read_csv(csv_file_path)

        report_time_in_minutes = int(config.get("reporting_time_in_min", 5))

        # Start Time and End Time from rainfall meta file
        rainfall_meta_file_name = config.get("rainfall_metadata_file")
        rainfall_meta_file_w_path = os.path.join(input_paths,
                                                 rainfall_meta_file_name)
        rainfall_meta_data = pd.read_csv(rainfall_meta_file_w_path, parse_dates=['start', 'end'])

        threads_per_simulation = config.get("num_threads_per_simulation", DEFAULT_THREADS_PER_SIMULATION)

        # Rainfall Path + Rainfall File name + .dat
        rainfall_hot_start_paths = paths['rainfall_hotstart_input']
        rg_file_type = ".dat"
        hs_file_type = ".hsf"

        input_file_lines = []
        with open(original_inp_file, 'r') as file:
            input_file_lines = file.readlines()

        hot_start_section_start_index = 0
        rain_gage_section_start_index = 0
        loading_section_index = 0
        options_section_index = 0

        for index, row in scenarios_table.iterrows():
            name = row['NAME']
            sdx = row['Standard_Deviation_X']
            sdy = row['Standard_Deviation_Y']
            wind_direction = row['Wind_Direction']
            rainfall_name = row['Rainfall_Event']
            new_inp_file = row['SWMM_InputFiles']

            if not os.path.exists(new_inp_file):
                x_value = row['Detonation_Location_X']
                y_value = row['Detonation_Location_Y']

                rainfall_hot_start_input_file_lines = input_file_lines.copy()

                hot_start_file = f"{rainfall_name}{hs_file_type}"
                hot_start_w_path = utils.return_path(
                    config_directory,
                    os.path.join(rainfall_hot_start_paths, hot_start_file)
                )
                hot_start_line = f'USE HOTSTART "{hot_start_w_path}\n'
                rainfall_hot_start_input_file_lines, hot_start_section_start_index = InputFileGenerator.replace_hot_start_section(
                    rainfall_hot_start_input_file_lines, hot_start_line, hot_start_section_start_index
                )

                rain_gage_file = f"{rainfall_name}{rg_file_type}"
                rain_gage_name = rain_gage_file.split(".")

                rainfall_file_w_path = utils.return_path(
                    config_directory,
                    os.path.join(rainfall_hot_start_paths, rain_gage_file)
                )
                rain_gage_line = f'{rg_name}\tFILE "{rainfall_file_w_path}'

                rainfall_hot_start_input_file_lines, rain_gage_section_start_index = InputFileGenerator.add_time_series_file(
                    rainfall_hot_start_input_file_lines, rain_gage_line, rain_gage_section_start_index
                )

                new_inp_file = os.path.join(
                    swmm_io_path,
                    f'{network_name}_{name}_{rain_gage_name[0]}_wind_direction_{wind_direction}_sd_{sdx}_{sdy}.inp'
                )

                scenarios_table.loc[index, "SWMM_InputFiles"] = new_inp_file
                scenarios_table.loc[index, "Scenario"] = Path(new_inp_file).stem

                # Get the start date and time and the end date and time based on the meta
                # data table and split the date and times for each.
                selected_row = rainfall_meta_data[rainfall_meta_data['rainfall'] == rainfall_name]
                start_date_time = selected_row.iloc[0]['start']
                end_date_time = selected_row.iloc[0]['end']

                start_rainfall_event_date = start_date_time.strftime('%m/%d/%Y')
                start_rainfall_event_time = start_date_time.strftime('%H:%M')

                end_rainfall_event_date = end_date_time.strftime('%m/%d/%Y')
                end_rainfall_event_time = end_date_time.strftime('%H:%M')

                options_dataframe = pd.DataFrame(
                    columns=['Name', 'Value'],
                    data=[
                        ['START_DATE', start_rainfall_event_date],
                        ['START_TIME', start_rainfall_event_time],
                        ['REPORT_START_DATE', start_rainfall_event_date],
                        ['REPORT_START_TIME', start_rainfall_event_time],
                        ['END_DATE', end_rainfall_event_date],
                        ['END_TIME', end_rainfall_event_time],
                        ['THREADS', threads_per_simulation],
                        ['REPORT_STEP', f"00:{report_time_in_minutes:02d}"],
                    ]
                )

                plume = GaussianPlume(
                    source_strength=contaminant_size_in_pico_curie,
                    source_location=(x_value, y_value),
                    direction=wind_direction,
                    standard_deviation=(sdx, sdy)
                )

                locations = scenarios_table[['X', 'Y']].values
                concentration_values = plume.concentration(locations=locations)
                scenarios_table['Pollutant'] = "Cesium"
                scenarios_table['Buildup'] = concentration_values

                # selected_columns = ['NAME', 'Pollutant', 'Buildup']
                # selected_df = scenario_table[selected_columns]

                rainfall_hot_start_input_file_lines, loading_section_index = InputFileGenerator.replace_loadings_section(
                    scenarios_table,
                    rainfall_hot_start_input_file_lines,
                    loading_section_index
                )

                rainfall_hot_start_input_file_lines, options_section_index = InputFileGenerator.replace_options_section(
                    options_dataframe,
                    rainfall_hot_start_input_file_lines,
                    options_section_index
                )

                with open(new_inp_file, 'w') as file:
                    file.writelines(rainfall_hot_start_input_file_lines)

        return scenarios_table
