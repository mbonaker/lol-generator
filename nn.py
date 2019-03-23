import logging
import os
from typing import TextIO

import numpy as np

import dataprovider
from config import ApplicationConfiguration

PURPOSE_TRAIN = 1
PURPOSE_TEST = 2
PURPOSE_PREDICT = 4


class NeuralNetwork:
    def __init__(self, data: dataprovider.DataProvider, config: ApplicationConfiguration):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data = data

        self.weights = None
        self.biases = None

    def load_weights(self, file_name: str) -> None:
        content = np.load(file_name)
        count = int(content['count'])
        for i in range(0, count):
            layer_name = "layer_{i:d}".format(i=i)
            biases_name = "biases_{i:d}".format(i=i)
            self.weights.append(content[layer_name])
            self.biases.append(content[biases_name])
        assert len(self.weights) == len(self.biases)

    def save_weights(self, file_name: str) -> None:
        weights = dict(("layer_{i:d}".format(i=i), layer) for i, layer in enumerate(self.weights))
        biases = dict(("biases_{i:d}".format(i=i), layer) for i, layer in enumerate(self.weights))
        assert len(weights) == len(biases)
        np.savez_compressed(file_name, count=len(self.weights), **weights, **biases)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.logger.log(logging.DEBUG, "Predict for known data of shape {shape!r}.".format(shape=x.shape))

        def relu(v: np.ndarray) -> np.ndarray:
            return np.maximum(v, 0)

        def sigmoid(v: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-v))

        x = x.T
        for layer, biases in zip(self.weights[:-1], self.biases[:-1]):
            x = relu(layer.T @ x + np.expand_dims(biases, 1))
        logit = (self.weights[-1] @ x + self.biases[-1]).T

        predictions = np.ndarray(shape=logit.shape, dtype=self.config.dtype)
        for csv_column in self.data.csv_structure.columns:
            full_np_slice = self.data.np_structure.csv_column_spec_to_np_slice(csv_column)
            np_slice = slice(full_np_slice.start - self.data.known.shape[1], full_np_slice.stop - self.data.known.shape[1])
            y_pred = logit[:, np_slice]
            if csv_column.handling == dataprovider.CsvColumnSpecification.HANDLING_ONEHOT or \
               csv_column.handling == dataprovider.CsvColumnSpecification.HANDLING_BOOL:
                predictions[:, np_slice] = sigmoid(y_pred)
            else:
                predictions[:, np_slice] = y_pred
        return predictions
