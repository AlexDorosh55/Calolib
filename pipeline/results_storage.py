import json
import os
import pathlib
from copy import copy
from typing import Dict, Union, Optional, IO, List

import pandas as pd


class Result:
    METRIC_VALUE = Union[float, int, str]

    def __init__(self):
        self._data = {
            'metrics': {}
        }

    def deserialize(self, file: IO) -> 'Result':
        self._data = json.load(file)
        return self

    def serialize(self, file: IO) -> None:
        return json.dump(self._data, file)

    @property
    def metrics(self) -> Dict[str, METRIC_VALUE]:
        return copy(self._data['metrics'])

    def add_metric(self, metric_name: str, value: METRIC_VALUE) -> None:
        self._data['metrics'][metric_name] = value


class ExperimentInfo:
    def __init__(self, dirpath: pathlib.Path, results_filename: str = 'results.json'):
        self._dirpath = dirpath
        self._results_filename = results_filename

    def _dump_result(self, result_filepath: pathlib.Path, result: Optional[Result] = None) -> None:
        if result is None:
            result = Result()
        with open(result_filepath, 'w') as file:
            result.serialize(file)

    def _get_result_filepath(self) -> pathlib.Path:
        result_filepath = self._dirpath / self._results_filename
        if not os.path.exists(result_filepath):
            self._dump_result(result_filepath)
        return result_filepath

    def get_result(self) -> Result:
        result_filepath = self._get_result_filepath()
        with open(result_filepath, 'r') as file:
            result = Result().deserialize(file)
        return result

    def save_result(self, result: Optional[Result] = None) -> None:
        result_filepath = self._get_result_filepath()
        self._dump_result(result_filepath, result)


class ResultsStorage:
    def __init__(self, storage_dir: str = './results',
                 results_filename: str = 'results.json'):
        self.storage_dir = pathlib.Path(storage_dir)
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        self.results_filename = results_filename

    def _init_exp_dir(self, exp_dirpath: pathlib.Path) -> None:
        pass

    def _get_exp_dirpath(self, exp_name: str) -> pathlib.Path:
        exp_name = exp_name.lower()
        exp_name = '_'.join(exp_name.split(' '))

        exp_dirpath = self.storage_dir / exp_name
        if not os.path.exists(exp_dirpath):
            os.mkdir(exp_dirpath)
            self._init_exp_dir(exp_dirpath)

        return exp_dirpath

    def get_experiment_info(self, exp_name: str) -> ExperimentInfo:
        exp_dirpath = self._get_exp_dirpath(exp_name)
        return ExperimentInfo(exp_dirpath, results_filename=self.results_filename)

    def compare_metrics_df(self, exp_names: List[str], metrics_names: List[str]) -> pd.DataFrame:
        rows = []

        for exp_name in exp_names:
            exp_info = self.get_experiment_info(exp_name)
            metrics = exp_info.get_result().metrics
            data_row = {
                metric_name: metrics.get(metric_name, None) for metric_name in metrics_names
            }
            data_row['name'] = exp_name

            rows.append(data_row)

        return pd.DataFrame(rows).set_index('name')
