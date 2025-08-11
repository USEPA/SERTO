import os
import os.path
import pathlib
import shutil
import time
from typing import Any
import glob
import json

import gc
import argparse

import mpi4py.MPI
import numpy as np
import pandas as pd
import yaml
import logging
from datetime import datetime, timezone
import chama

from mpi4py import MPI
import getmac

from ... import SERTODefaults
from . import , SensorGenerator, SimulationsRunner
from .. import utils


def run(config: str, **kwargs: Any):
    t0 = time.time()

    """Run the optimizer."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    current_datetime = datetime.now(timezone.utc)
    current_datetime_path = current_datetime.strftime('%Y-%m-%d_%H-%M-%S%Z')

    config = os.path.abspath(config)

    # Set up logging to file with same name as configuration file
    log_file = config.replace('.yaml', '.log').replace('.yml', '.log')

    if os.path.exists(log_file):
        shutil.copy(log_file, log_file + f'.{current_datetime_path}.bak')

    logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w')

    # Log computer name ahd mac address
    computer_name = getmac.get_mac_address()
    logging.info(f'Computer: {MPI.Get_processor_name()} | Rank: {rank} | MAC: {computer_name}')

    if rank == 0:

        # Log the start time
        current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
        current_datetime_path = current_datetime.strftime('%Y-%m-%d_%H-%M-%S%Z')
        logging.info(f'Start Time: {current_datetime_str}')

        # Get the total number of processors and the rank of this processor
        num_procs = comm.Get_size()
        logging.info(f'Number of processors: {num_procs}')

        # Read the configuration file
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
            logging.info(f'Configuration: {config_data}')

            # configure summary tables and plots

            io_paths = config_data['paths']
            ensemble_path = io_paths['ensemble_runs']
            config_directory = str(pathlib.Path(config).parent.absolute())
            ensemble_path = utils.return_path(config_directory, ensemble_path)
            os.makedirs(ensemble_path, exist_ok=True)

            # configure output files path
            swmm_io_path = utils.return_path(config_directory, io_paths['inp_output'])
            os.makedirs(swmm_io_path, exist_ok=True)

            chama_signals_path = utils.return_path(config_directory, io_paths['signal_output'])
            os.makedirs(chama_signals_path, exist_ok=True)

            chama_io_path = utils.return_path(config_directory, io_paths['output_dir'])
            os.makedirs(chama_io_path, exist_ok=True)

            # Configure all swmm input files
            logging.info(f'Creating Ensemble SWMM input files')
            delete_results_after_run = config_data.get('delete_results_after_run', True)
            use_existing = config_data.get('use_existing_results', False)
            use_existing = kwargs.get('use_existing', use_existing)

            scenario_input_matrix_table_path = os.path.join(ensemble_path, 'scenario_input_matrix_table.csv')

            if use_existing and os.path.exists(scenario_input_matrix_table_path):
                scenario_input_matrix_table = pd.read_csv(scenario_input_matrix_table_path, index_col=0)
                scenario_input_matrix_table = InputFileGenerator.create_input_files_from_table(
                    config=config_data,
                    config_directory=config_directory,
                    swmm_io_path=swmm_io_path,
                    scenarios_table=scenario_input_matrix_table,
                )
                scenario_input_matrix_table.to_csv(scenario_input_matrix_table_path)
            else:
                scenario_input_matrix_table = InputFileGenerator.create_input_files(
                    config=config_data,
                    config_directory=config_directory,
                    swmm_io_path=swmm_io_path,
                )
                scenario_input_matrix_table.to_csv(scenario_input_matrix_table_path)

            scenarios_dataframe = pd.DataFrame(
                {
                    'Scenario': scenario_input_matrix_table['Scenario'].values,
                    'Undetected_Impact': config_data.get('undetected_impact', 0),
                    'Probability': 1 / len(scenario_input_matrix_table),
                }
            )

            logging.info(f'Finished Creating Ensemble SWMM input files')

            # Generate possible sensor locations
            logging.info(f'Generating possible sensor locations')
            sensor_locations = SensorGenerator.generate_possible_sensor_locations(
                config=config_data,
                config_directory=config_directory,
            )
            sensor_locations.to_csv(os.path.join(chama_io_path, 'possible_sensor_locations.csv'))
            logging.info(f'Finished generating possible sensor locations')

            logging.info(f'Running ensemble and generating signals')
            runner = SimulationsRunner(
                config=config_data,
                scenario_input_matrix_table=scenario_input_matrix_table,
                sensor_locations=sensor_locations,
                num_processors=num_procs,
                use_existing=use_existing,
                delete_results_after_run=delete_results_after_run
            )
            results = runner.execute(mpi_comm=comm)
            results.to_csv(os.path.join(chama_io_path, 'ensemble_results.csv'))
            logging.info(f'Finished running ensemble and generating signals')

            logging.info(f'Combining signal files')

            gc.collect()
            combined_signals_filepath = os.path.join(chama_signals_path, 'combined_signals.csv')
            if use_existing and os.path.exists(combined_signals_filepath):
                combined_signals = pd.read_csv(combined_signals_filepath, index_col=0)
            else:
                combined_signals = SimulationsRunner.combine_chama_signals(results)
                combined_signals.to_csv(os.path.join(chama_io_path, 'combined_signals.csv'))

            logging.info(f'Finished combining signal files')

            logging.info(f'Running CHAMA optimization')

            detector_threshold = config_data.get('detector_threshold')
            min_time = combined_signals['T'].min()
            max_time = combined_signals['T'].max()
            sample_times = np.arange(min_time, max_time + 1)

            positions = [chama.sensors.Stationary(location=sensor) for sensor in sensor_locations['Sensor'].values]
            detector_points = [
                chama.sensors.Point(threshold=detector_threshold, sample_times=sample_times)
                for sensor in range(sensor_locations.shape[0])
            ]
            sensors = {
                position.location: chama.sensors.Sensor(position=position, detector=detector_points[i])
                for i, position in enumerate(positions)
            }

            gc.collect()
            logging.info(f'Computing detection times')
            detection_times = chama.impact.extract_detection_times(
                signal=combined_signals,
                sensors=sensors,
            )
            detection_times.to_csv(os.path.join(chama_io_path, 'detection_times.csv'))
            logging.info(f'Finished computing detection times')

            logging.info(f'Computing detection times stats')
            detection_times_stats = chama.impact.detection_time_stats(detection_times)
            detection_times_stats.to_csv(os.path.join(chama_io_path, 'detection_times_stats.csv'))
            logging.info(f'Finished computing detection times stats')

            min_det_time = detection_times_stats[['Scenario', 'Sensor', 'Min']]
            min_det_time = min_det_time.rename(columns={'Min': 'Impact'})
            sensor_locations['Cost'] = sensor_locations['Cost'].astype(float)

            formulation = config_data.get('formulation', 'coverage')
            sensor_budget = config_data.get('sensor_budget')

            if formulation == 'impact':
                impact_formulation = chama.optimize.ImpactFormulation()
                results = impact_formulation.solve(
                    impact=min_det_time,
                    sensor_budget=sensor_budget,
                    sensor=sensor_locations,
                    scenario=scenarios_dataframe,
                    use_scenario_probability=True,
                    use_sensor_cost=True,
                )

            else:
                scenario_time, new_scenario = chama.impact.detection_times_to_coverage(
                    detection_times=detection_times,
                    coverage_type=config_data.get('opt_approach', 'scenario-time'),
                    scenario=scenarios_dataframe
                )

                scenario_time.to_csv(os.path.join(chama_io_path, 'coverage_results.csv'))
                new_scenario.to_csv(os.path.join(chama_io_path, 'new_scenario_results.csv'))

                new_scenario = scenarios_dataframe.rename(columns={
                    'Scenario': 'Entity',
                    'Probability': 'Weight'
                })

                coverage_formulation = chama.optimize.CoverageFormulation()
                results = coverage_formulation.solve(
                    coverage=scenario_time,
                    sensor_budget=sensor_budget,
                    sensor=sensor_locations,
                    entity=new_scenario,
                    use_sensor_cost=True,
                )

            # with open(os.path.join(chama_io_path, 'optimization_results.yml'), 'w') as results_file:
            #     yaml.dump(results, results_file)

            with open(os.path.join(chama_io_path, 'optimization_results.json'), 'w') as results_file:
                json.dump(results, results_file, indent=4)

        t1 = time.time()
        logging.info(f'Total time: {t1 - t0} seconds')
        logging.info(f'Finished running CHAMA optimization')

    else:

        logging.info(f'Rank: {rank}')

        import multiprocessing as mp

        status = mpi4py.MPI.Status()
        comm.probe(source=0, tag=0, status=status)
        buffer = bytearray(b" " * (status.count + 1))
        data = comm.irecv(buf=buffer, source=0, tag=0).wait()

        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)

            io_paths = config_data['paths']
            config_directory = str(pathlib.Path(config).parent.absolute())

            # configure output files path
            chama_io_path = utils.return_path(config_directory, io_paths['output_dir'])
            num_threads_per_simulation = config_data.get('num_threads_per_simulation', 4)
            max_concurrent_processes = min(int(mp.cpu_count() / num_threads_per_simulation), mp.cpu_count())

            sensor_locations = pd.read_csv(os.path.join(chama_io_path, 'possible_sensor_locations.csv'), index_col=0)

            pollutant = config_data.get('pollutant')
            delete_results_after_run = config_data.get('delete_results_after_run', True)
            use_existing = config_data.get('use_existing_results', False)
            use_existing = kwargs.get('use_existing', use_existing)

            results = SimulationsRunner.run_all(
                inp_files=data.tolist(),
                max_concurrent_processes=max_concurrent_processes,
                potential_sensor_locations=sensor_locations,
                pollutant=pollutant,
                delete_results_after_run=delete_results_after_run,
                use_existing=use_existing,
                prefix=f"Processor {rank}:"
            )

            req = comm.isend(results, dest=0, tag=0)
            req.wait()
