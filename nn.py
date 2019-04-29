import logging
import os
import time
from typing import TextIO

import numpy as np

import dataprovider as dp
from config import ApplicationConfiguration

PURPOSE_TRAIN = 1
PURPOSE_TEST = 2
PURPOSE_PREDICT = 4


class NeuralNetwork:
    def __init__(self, field_structure: dp.FieldStructure, column_structure: dp.ColumnStructure, config: ApplicationConfiguration):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.field_structure = field_structure
        self.column_structure = column_structure

        self.weights = None
        self.biases = None

    def load_weights(self, file_name: str) -> None:
        content = np.load(file_name)
        count = int(content['count'])
        self.weights = []
        self.biases = []
        for i in range(0, count):
            layer_name = "layer_{i:d}".format(i=i)
            biases_name = "biases_{i:d}".format(i=i)
            weights = content[layer_name]
            biases = content[biases_name]
            assert weights.shape[1] == biases.shape[0]
            assert biases.shape[1] == 1
            self.weights.append(weights)
            self.biases.append(biases)
        assert len(self.weights) == len(self.biases)

    def save_weights(self, file_name: str) -> None:
        weights = dict(("layer_{i:d}".format(i=i), layer) for i, layer in enumerate(self.weights))
        biases = dict(("biases_{i:d}".format(i=i), layer) for i, layer in enumerate(self.biases))
        assert len(weights) == len(biases)
        np.savez_compressed(file_name, count=len(self.weights), **weights, **biases)


class Generator(NeuralNetwork):

    def __init__(self, field_structure: dp.FieldStructure, column_structure: dp.ColumnStructure, config: ApplicationConfiguration):
        super().__init__(field_structure, column_structure, config)
        unknown_field_structure = dp.FieldStructure.make(self.field_structure.data_path, dp.PORTION_UNKNOWN - dp.PORTION_WIN)
        unknown_column_structure = dp.ColumnStructure.make(unknown_field_structure, self.config.dtype, dp.PORTION_UNKNOWN - dp.PORTION_WIN)
        self.handling_slices = list(unknown_column_structure.generate_handling_slices(self.config.ignored_columns))

    def predict(self, x: np.ndarray, alpha=0.2) -> np.ndarray:
        self.logger.log(logging.DEBUG, "Predict for known data of shape {shape!r}.".format(shape=x.shape))
        time_start = time.perf_counter()

        def leaky_relu(v: np.ndarray) -> np.ndarray:
            return np.where(v > 0, v, v * alpha)

        x = x.T
        for layer, biases in zip(self.weights[:-1], self.biases[:-1]):
            x = leaky_relu(layer.T @ x + biases)
        logit = (self.weights[-1].T @ x + self.biases[-1]).T

        if self.config.ignored_columns:
            predictions = np.ndarray(shape=logit.shape, dtype=self.config.dtype)
            for data_slice, handling in self.handling_slices:
                y_pred = logit[:, data_slice]
                if handling == dp.FieldSpecification.HANDLING_NONE:
                    predictions[:, data_slice] = np.nan
                else:
                    predictions[:, data_slice] = y_pred
        else:
            predictions = logit

        self.logger.log(logging.DEBUG, "Prediction done for {shape!r} in {duration:.7f} seconds.".format(shape=x.shape, duration=time.perf_counter() - time_start))
        return predictions


class WinEstimator(NeuralNetwork):

    def __init__(self, field_structure: dp.FieldStructure, column_structure: dp.ColumnStructure, config: ApplicationConfiguration):
        super().__init__(field_structure, column_structure, config)
        input_fields = dp.FieldStructure.make(self.field_structure.data_path, dp.PORTION_INTERESTING - dp.PORTION_WIN)
        input_columns = dp.ColumnStructure.make(input_fields, self.config.dtype, dp.PORTION_INTERESTING - dp.PORTION_WIN)
        self.handling_slices = input_columns.generate_handling_slices(self.config.ignored_columns)

    def predict(self, x: np.ndarray, alpha=0.2) -> np.ndarray:
        self.logger.log(logging.DEBUG, "Predict for interesting data of shape {shape!r}.".format(shape=x.shape))
        time_start = time.perf_counter()

        def leaky_relu(v: np.ndarray) -> np.ndarray:
            return np.where(v > 0, v, v * alpha)

        x = x.T
        for layer, biases in zip(self.weights[:-1], self.biases[:-1]):
            x = leaky_relu(layer.T @ x + biases)
        logit = (self.weights[-1].T @ x + self.biases[-1]).T

        if self.config.ignored_columns:
            predictions = np.ndarray(shape=logit.shape, dtype=self.config.dtype)
            for data_slice, handling in self.handling_slices:
                y_pred = logit[:, data_slice]
                if handling == dp.FieldSpecification.HANDLING_NONE:
                    predictions[:, data_slice] = np.nan
                else:
                    predictions[:, data_slice] = y_pred
        else:
            predictions = logit

        self.logger.log(logging.DEBUG, "Prediction done for {shape!r} in {duration:.7f} seconds.".format(shape=x.shape, duration=time.perf_counter() - time_start))
        return predictions


class Discriminator(NeuralNetwork):
    def predict(self) -> np.ndarray:
        raise NotImplementedError()
