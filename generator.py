#!/usr/bin/env python3
import logging
import logging.config
import sys

import tensorflow as tf
import numpy as np
import yaml

import dataprovider
from nn import NeuralNetwork
from config import ApplicationConfiguration


def handle_all_inputs(config: ApplicationConfiguration):
    corpus = dataprovider.CorpusProvider("../data", np.dtype(np.float16))

    if config.should_train:
        nn = NeuralNetwork(corpus, config)
        with tf.Session() as session:
            nn.start_session(session)
            for i in range(0, config.steps):
                nn.train(session)
                if i % 100 == 0:
                    nn.train_eval(session)
                if i % 1000 == 0:
                    nn.test_eval(session)
            nn.stop_session()

    if config.should_read_stdin:
        in_data = dataprovider.KnownStdinProvider("../data", np.dtype(np.float16))
        out_data = dataprovider.DataProvider("../data", np.dtype(np.float16), dataprovider.PORTION_INTERESTING - dataprovider.PORTION_WIN)
        out_data.create_nan_data(in_data.known.shape[0])
        out_data.known = in_data.known
        out_data.write_as_csv(sys.stdout)


if __name__ == '__main__':
    with open("logging.yaml", 'rt') as logging_yaml_file:
        logging.config.dictConfig(yaml.safe_load(logging_yaml_file.read()))
    handle_all_inputs(ApplicationConfiguration(sys.argv[1:]))
