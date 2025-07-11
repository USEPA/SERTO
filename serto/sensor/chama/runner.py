import logging
import os
from typing import List, Dict, Any, Iterable, Tuple
from functools import partial
import re
import time
from pathlib import Path

import pandas as pd
from swmm.toolkit import solver
from pyswmm import Output
from swmm.toolkit import output, shared_enum

import multiprocessing as mp
from threading import Lock
from mpi4py import MPI
import numpy as np

from .. import DEFAULT_THREADS_PER_SIMULATION
from .. import utils


class SimulationsRunner:
    def __init__(
            self,
            config: Dict[Any, Any],
            scenario_input_matrix_table: pd.DataFrame,
            sensor_locations: pd.DataFrame,
            num_processors: int,
            delete_results_after_run: bool = True,
            use_existing: bool = False,
            **kwargs
    ):

        self._isRunning = False
        self._progress = 0.0
        self._inp_files = scenario_input_matrix_table
        self._model_run_results = pd.DataFrame(
            columns=[
                'inp_file',
                'sim_start_time',
                'sim_end_time',
                'sim_duration',
                'sim_status',
                'sim_error',
                'runoff_continuity_error',
                'hydration_continuity_error',
                'quality_continuity_error',
                'percent_not_converged',
            ]
        )

        self.num_processors = num_processors
        self.pollutant = config.get('pollutant')
        self.threads_per_simulation = config.get('threads_per_simulation', DEFAULT_THREADS_PER_SIMULATION)
        self.potential_sensor_locations = sensor_locations
        self.use_existing = use_existing
        self.delete_results_after_run = delete_results_after_run

    @property
    def isRunning(self):
        return self._isRunning

    @property
    def isBusy(self):
        return self._isRunning

    @property
    def progress(self):
        return self._progress

    @property
    def input_files(self):
        return self._inp_files

    @property
    def results(self):
        return self._model_run_results

    @staticmethod
    def generate_report_summary(rpt_file) -> Dict[str, Any]:
        """
        Read the summary information from the report file
        :param rpt_file:
        :return:
        """
        data = {
            'error': '',
        }

        current_section = None

        with open(rpt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Runoff Quantity Continuity'):
                    current_section = 'runoff_continuity_error'
                elif line.startswith('Runoff Quality Continuity'):
                    current_section = 'runoff_quality_continuity_error'
                elif line.startswith('Flow Routing Continuity'):
                    current_section = 'flow_routing_continuity_error'
                elif line.startswith('Quality Routing Continuity'):
                    current_section = 'quality_routing_continuity_error'
                elif current_section == 'runoff_continuity_error' and line.startswith('Continuity Error (%)'):
                    data['runoff_continuity_error'] = line.split()[-1]
                elif current_section == 'runoff_quality_continuity_error' and line.startswith('Continuity Error (%)'):
                    data['runoff_quality_continuity_error'] = line.split()[-1]
                elif current_section == 'flow_routing_continuity_error' and line.startswith('Continuity Error (%)'):
                    data['flow_routing_continuity_error'] = line.split()[-1]
                elif current_section == 'quality_routing_continuity_error' and line.startswith('Continuity Error (%)'):
                    data['quality_routing_continuity_error'] = line.split()[-1]
                elif line.startswith('% of Steps Not Converging'):
                    data['percent_not_converged'] = line.split()[-1]
                elif line.startswith('Analysis begun on:'):
                    data['sim_start_time'] = line.replace('Analysis begun on:', '').strip()
                elif line.startswith('Analysis ended on:'):
                    data['sim_end_time'] = line.replace('Analysis ended on:', '').strip()
                elif line.startswith('Total elapsed time:'):
                    data['sim_duration'] = line.replace('Total elapsed time:', '').strip()
                    break
                elif line.startswith('ERROR'):
                    data['error'] = line
        return data

    @staticmethod
    def generate_chama_signal(out_file, potential_sensor_locations: pd.DataFrame, pollutant: str) -> pd.DataFrame:

        results = []

        with Output(out_file) as output:
            scenario_name = Path(out_file).stem
            for index, row in potential_sensor_locations.iterrows():
                node_id = row['Sensor']
                pollutant_series = output.node_series(
                    node_id,
                    shared_enum.NodeAttribute.POLLUT_CONC_0,
                )
                pollutant_series = pd.DataFrame.from_dict(pollutant_series, orient='index', columns=[pollutant])

                pollutant_series['Node'] = node_id
                pollutant_series['T'] = np.arange(0, len(pollutant_series), 1)
                pollutant_series = pollutant_series[['Node', 'T', pollutant]]
                pollutant_series.rename(columns={pollutant: scenario_name}, inplace=True)
                pollutant_series.reset_index(drop=True, inplace=True)
                results.append(pollutant_series)

        combined_results = pd.concat(results, axis='rows')

        return combined_results

    @staticmethod
    def combine_chama_signals(input_files_matrix: pd.DataFrame) -> pd.DataFrame:
        import gc
        combined_results = None

        for index, row in input_files_matrix.iterrows():
            signal_file = row['signals_file']
            if os.path.exists(signal_file):
                signal_file = pd.read_csv(signal_file, index_col=0)
                if combined_results is None:
                    combined_results = signal_file
                else:
                    combined_results = pd.merge(left=combined_results, right=signal_file, on=['Node', 'T'], how='outer')

                gc.collect()

        return combined_results

    @staticmethod
    def run(
            inp_file: str,
            potential_sensor_locations: pd.DataFrame,
            pollutant: str,
            use_existing: bool = False,
            delete_results_after_run: bool = True,
            **kwargs
    ) -> Tuple[int, Dict[str, Any]]:
        """
        :param use_existing:
        :param potential_sensor_locations:
        :param pollutant:
        :param delete_results_after_run:
        :param inp_file:
        :param kwargs:
        :return:
        """

        out_file = inp_file.replace('.inp', '.out')
        rpt_file = inp_file.replace('.inp', '.rpt')

        error = 0
        report_summary = {
            'runoff_continuity_error': 0.0,
            'signals_file': '',
            'error': ''
        }

        # return 0, report_summary

        if use_existing and os.path.exists(out_file) and os.path.exists(rpt_file):
            logging.info(f'Using existing files: {out_file}, {rpt_file}')
            error = 0
        else:
            logging.info(f'Running {inp_file}')

            try:
                error = solver.swmm_run(inp_file, rpt_file, out_file)
            except Exception as e:
                logging.error(f'Error running {inp_file}: {e}')
                report_summary['error'] = str(e)
                error = 1

        if error is None or error == 0:
            # with semaphore:
            report_summary = SimulationsRunner.generate_report_summary(rpt_file)
            signal_path = out_file.replace('.out', '_signals.csv')

            if use_existing and os.path.exists(signal_path):
                pass
            else:
                signal = SimulationsRunner.generate_chama_signal(out_file, potential_sensor_locations, pollutant)
                signal.to_csv(out_file.replace('.out', '_signals.csv'))

            report_summary['signals_file'] = signal_path

            if delete_results_after_run:
                os.remove(out_file)
                os.remove(rpt_file)

            error = 0

        else:
            logging.error(f'Error running {inp_file}')
            return error, report_summary

        return error, report_summary

    @staticmethod
    def run_all(
            inp_files: List[str],
            max_concurrent_processes: 8,
            potential_sensor_locations: pd.DataFrame,
            pollutant: str,
            delete_results_after_run: bool = True,
            use_existing: bool = False,
            prefix: str = 'Progress:',
            **kwargs
    ):
        """

        :param prefix:
        :param use_existing:
        :param delete_results_after_run:
        :param pollutant:
        :param potential_sensor_locations:
        :param inp_files:
        :param max_concurrent_processes:
        :param kwargs:
        :return:
        """
        run_partial = partial(
            SimulationsRunner.run,
            potential_sensor_locations=potential_sensor_locations,
            pollutant=pollutant,
            delete_results_after_run=delete_results_after_run,
            use_existing=use_existing
        )

        results = []
        with mp.Pool(processes=max_concurrent_processes) as executor:
            test_results = executor.imap(run_partial, inp_files)
            for i, result in enumerate(test_results):
                utils.progress_bar(
                    iteration=i + 1,
                    total=len(inp_files),
                    prefix=prefix,
                    suffix='Complete'
                )
                results.append(result)

            executor.close()
            executor.join()

        return results

    def execute(self, mpi_comm) -> pd.DataFrame:

        # Partition job over all processors and cores to be evenly distributed
        self._inp_files['SuccessfulRun'] = False
        self._inp_files['ProcessorRank'] = 0

        partitioned_runs = np.array_split(self._inp_files, self.num_processors)

        self._isRunning = True
        proc = 1
        runs = {}
        for i, partition in enumerate(partitioned_runs):
            if len(partitioned_runs) > 1 and i < len(partitioned_runs) - 1:
                logging.info(f"Sending data: {partition['SWMM_InputFiles']} to processor {proc}")
                data_array = partition['SWMM_InputFiles'].values
                req = mpi_comm.isend(data_array, dest=proc)
                req.wait()
                runs[proc] = partition
                proc += 1
            else:
                max_concurrent_processes = min(int(mp.cpu_count() / self.threads_per_simulation), mp.cpu_count())
                model_run_results = SimulationsRunner.run_all(
                    inp_files=partition['SWMM_InputFiles'].values,
                    max_concurrent_processes=max_concurrent_processes,
                    potential_sensor_locations=self.potential_sensor_locations,
                    pollutant=self.pollutant,
                    use_existing=self.use_existing,
                    delete_results_after_run=self.delete_results_after_run,
                    prefix=f"Processor {0}:"
                )

                for i in range(partition.shape[0]):
                    self._inp_files.loc[partition.index[i], 'SuccessfulRun'] = model_run_results[i][0]
                    results_report = model_run_results[i][1]
                    for key, value in results_report.items():
                        self._inp_files.loc[partition.index[i], key] = value

        logging.info("Waiting for all processors to finish")
        status = MPI.Status()

        for i in range(1, self.num_processors):
            mpi_comm.probe(source=i, tag=0, status=status)
            buffer = bytearray(b" " * (status.count + 1))
            t = mpi_comm.irecv(buf=buffer, source=i).wait()
            logging.info(f"Received data from processor {i}")
            partition = runs[i]

            for i in range(partition.shape[0]):
                partition_index = partition.index[i]
                self._inp_files.loc[partition_index, 'SuccessfulRun'] = t[i][0]
                self._inp_files.loc[partition_index, 'ProcessorRank'] = i

                results_report = t[i][1]
                for key, value in results_report.items():
                    self._inp_files.loc[partition_index, key] = value

        self._isRunning = False
        self._progress = 100.0

        return self._inp_files
