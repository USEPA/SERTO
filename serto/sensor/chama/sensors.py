# python imports
from typing import List
import re

# third-party imports
import pandas as pd
import os

# local imports
from ... import SERTODefaults


class SensorGenerator:
    def __init__(self):
        pass

    @staticmethod
    def read_section(section_name, content: List[str]) -> List[str]:

        start_tag = f'[{section_name}]'
        tag_found = False

        items = []

        for line in content:

            stripped_line = line.strip()

            if stripped_line == start_tag:
                tag_found = True
            elif tag_found and stripped_line.startswith('['):
                break
            elif tag_found and not (stripped_line.startswith(';') or stripped_line == ''):
                split_values = re.split(r'\t|\s+', stripped_line)
                items.append(split_values[0])

        return items

    @staticmethod
    def generate_possible_sensor_locations(
            config,
            config_directory,
    ) -> pd.DataFrame:

        base_inp_name = utils.return_path(config_directory, config.get('inp_file'))

        sensor_objects = config.get("sensor_objects")

        sensor_array_upper = [obj.upper() for obj in sensor_objects]
        section_dfs = []

        with open(base_inp_name, 'r') as file:

            content = file.readlines()

            # Read each section and store it in the dictionary
            for section in sensor_array_upper:
                section_dfs.extend(SensorGenerator.read_section(section, content))
        # Combine first columns into a new DataFrame
        sensor_cost_df = pd.DataFrame({
            'Sensor': section_dfs,
            'Cost': 1.0,
        })

        # Print the result
        return sensor_cost_df