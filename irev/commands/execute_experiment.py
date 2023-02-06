import importlib
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from irev.algorithms import *
from irev.metrics import *
from irev.enviroment import split


def get_class(dynamic_imported_modules, class_name: str):
    try:
        return globals()[class_name]
    except KeyError:
        for module in dynamic_imported_modules:
            try:
                return getattr(module, class_name)
            except:
                continue

    raise Exception(f"Class not found: {class_name}.")

def load_dataset(path: str) -> Tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(path)

    U = df["user_id"].max() + 1
    I = df["item_id"].max() + 1

    return (df, U, I)

def execute_experiment(args):
    modules = list()

    for module_name in args.extra_modules:
        modules.append(importlib.import_module(module_name))

    with open(args.config_file, 'r') as f:
        experiment_config = yaml.load(f, Loader=yaml.BaseLoader)

    for dataset_name in args.datasets:
        dataset_config = experiment_config["Data"][dataset_name]

        df, U, I = load_dataset(dataset_config["path"])

        split_config = dataset_config["Split"]

        train, test = split.user_based_split(df, train_size=float(split_config["train_size"]))

        for algorithm_class_name in args.algorithms:
            algorithm_parameters = experiment_config["Algorithms"][algorithm_class_name]

            algorithm: BaseAlgorithm = get_class(modules, algorithm_class_name)(**algorithm_parameters)

            prediction_matrix = algorithm.fit(train, U, I)

            # TODO: Save executions.

            for metric_class_name in args.metrics:
                metric_class: BaseMetric = get_class(modules, metric_class_name)

                metric_parameters = experiment_config["Metrics"][metric_class_name]

                print(f'{metric_class_name}: {metric_class.calculate(prediction_matrix, train, test, **metric_parameters)}')
