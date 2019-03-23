import logging
from io import TextIOWrapper
from time import perf_counter
from typing import Optional

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

    def load_weights(self, file: TextIOWrapper) -> None:
        content = np.load(file)
        count = int(content['count'])
        for i in range(0, count):
            layer_name = "layer_{i:d}".format(i=i)
            biases_name = "boases_{i:d}".format(i=i)
            self.weights.append(content[layer_name])
            self.biases.append(content[biases_name])
        assert len(self.weights) == len(self.biases)

    def save_weights(self, file: TextIOWrapper) -> None:
        weights = dict(("layer_{i:d}".format(i=i), layer) for i, layer in enumerate(self.weights))
        biases = dict(("biases_{i:d}".format(i=i), layer) for i, layer in enumerate(self.weights))
        assert len(weights) == len(biases)
        np.savez_compressed(file, count=len(self.weights), **weights, **biases)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.logger.log(logging.DEBUG, "Predict for known data of shape {shape!r}.".format(shape=x.shape))
        relu = lambda x: np.maximum(x, 0)
        features = x
        x = x.T
        for layer, biases in zip(self.weights, self.biases):
            x = relu(layer.T @ x + np.expand_dims(biases, 1))
        logit = tf.transpose(tf.add(tf.matmul(w, x, True), b), name="logit")
        raise NotImplementedError()
