#!/usr/bin/env python3
import logging
import logging.config
import sys

import numpy as np
import yaml

import dataprovider
from nn import Generator
from config import ApplicationConfiguration


def handle_all_inputs(config: ApplicationConfiguration):
    logger = logging.getLogger("generator")

    if config.should_train:
        corpus = dataprovider.CorpusProvider("../data", np.dtype(np.float16), known_data_is_optional=True)
        import tfnn
        nn = tfnn.TrainableGenerator(corpus.fields, corpus.columns, config)
        if config.try_consistently:
            done = False
            while not done:
                try:
                    nn.train("../data/generator_state_{label:s}.npz".format(label=str(config)), corpus)
                    done = True
                except:
                    logger.exception("Error during training, but try consistently...", )
                    done = False
        else:
            nn.train("../data/generator_state_{label:s}.npz".format(label=str(config)), corpus)

    if config.shoud_optimize_hyperparameters:
        corpus = dataprovider.CorpusProvider("../data", np.dtype(np.float16), known_data_is_optional=True)
        import confopt
        opt = confopt.ConfigurationOptimizer(corpus)
        opt.optimize()

    if config.should_read_stdin:
        in_data = dataprovider.KnownStdinProvider("../data", np.dtype(np.float16), True)
        out_data = dataprovider.DataProvider("../data", np.dtype(np.float16), dataprovider.PORTION_INTERESTING - dataprovider.PORTION_WIN, True)
        out_data.create_nan_data(in_data.known.shape[0])
        nn = Generator(in_data.fields, in_data.columns, config)
        out_data.known = in_data.known
        nn.load_weights("../data/generator_state_{label:s}.npz".format(label=str(config)))
        out_data.unknown_without_win = nn.predict(in_data)

        out_data.write_as_csv(sys.stdout)


if __name__ == '__main__':
    with open("logging.yaml", 'rt') as logging_yaml_file:
        logging.config.dictConfig(yaml.safe_load(logging_yaml_file.read()))
    handle_all_inputs(ApplicationConfiguration(sys.argv[1:]))
