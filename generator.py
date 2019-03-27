#!/usr/bin/env python3
import logging
import logging.config
import sys

import numpy as np
import yaml

import dataprovider
from nn import NeuralNetwork
from config import ApplicationConfiguration
import config as cfg


# called as: gen_output, gen_vars, gen_loss = generator.get_trainable_output(batch_size)
def get_trainable_output(batch_size: int):
    config = ApplicationConfiguration("")
    config.set(cfg.LABEL, 'adversarial')
    config.set(cfg.CORPUS_FILE_NAME, 'Matches-orig')
    config.set(cfg.BATCH_SIZE, batch_size)
    corpus = dataprovider.CorpusProvider("../data", np.dtype(np.float16), portion=dataprovider.PORTION_INTERESTING, known_data_is_optional=False, corpus_file_name=config.corpus_file_name)
    import tfnn
    nn = tfnn.TrainableNeuralNetwork(corpus, config, promise_batch_size=True)
    return nn



def handle_all_inputs(config: ApplicationConfiguration):

    if config.should_train:
        corpus = dataprovider.CorpusProvider("../data", np.dtype(np.float16), known_data_is_optional=True, corpus_file_name=config.corpus_file_name)
        import tfnn
        nn = tfnn.TrainableNeuralNetwork(corpus, config)
        nn.train("../data/generator_state_{label:s}.npz".format(label=str(config)))

    if config.should_read_stdin:
        in_data = dataprovider.KnownStdinProvider("../data", np.dtype(np.float16), True)
        out_data = dataprovider.DataProvider("../data", np.dtype(np.float16), dataprovider.PORTION_INTERESTING - dataprovider.PORTION_WIN, True)
        out_data.create_nan_data(in_data.known.shape[0])
        nn = NeuralNetwork(in_data, config)
        out_data.known = in_data.known
        nn.load_weights("../data/generator_state_{label:s}.npz".format(label=str(config)))
        out_data.unknown_without_win = nn.predict()

        out_data.write_as_csv(sys.stdout)


if __name__ == '__main__':
    with open("logging.yaml", 'rt') as logging_yaml_file:
        logging.config.dictConfig(yaml.safe_load(logging_yaml_file.read()))
    handle_all_inputs(ApplicationConfiguration(sys.argv[1:]))
