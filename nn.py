import logging
import os
from typing import TextIO

import numpy as np

import dataprovider as dp
from config import ApplicationConfiguration

PURPOSE_TRAIN = 1
PURPOSE_TEST = 2
PURPOSE_PREDICT = 4


class NeuralNetwork:
    def __init__(self, data: dp.DataProvider, config: ApplicationConfiguration):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data = data

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

    def predict(self) -> np.ndarray:
        unknown_csv_structure = dp.CsvCorpusStructure(self.data.data_path, dp.PORTION_UNKNOWN - dp.PORTION_WIN)
        unknown_data_structure = dp.NumpyCorpusStructure(unknown_csv_structure, self.config.dtype, dp.PORTION_UNKNOWN - dp.PORTION_WIN)

        x = self.data.get_ndarray(dp.PORTION_KNOWN)
        self.logger.log(logging.DEBUG, "Predict for known data of shape {shape!r}.".format(shape=x.shape))

        def relu(v: np.ndarray) -> np.ndarray:
            return np.maximum(v, 0)

        def sigmoid(v: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-v))

        x = x.T
        for layer, biases in zip(self.weights[:-1], self.biases[:-1]):
            x = relu(layer.T @ x + biases)
        logit = (self.weights[-1].T @ x + self.biases[-1]).T

        predictions = np.ndarray(shape=logit.shape, dtype=self.config.dtype)
        for data_slice, handling in unknown_data_structure.generate_handling_slices(self.config.ignored_columns):
            y_pred = logit[:, data_slice]
            if handling == dp.CsvColumnSpecification.HANDLING_ONEHOT or \
               handling == dp.CsvColumnSpecification.HANDLING_BOOL:
                predictions[:, data_slice] = sigmoid(y_pred)
            elif handling == dp.CsvColumnSpecification.HANDLING_NONE:
                predictions[:, data_slice] = np.nan
            else:
                predictions[:, data_slice] = y_pred
        return predictions
