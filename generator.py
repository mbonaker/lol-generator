#!/usr/bin/env python3
import logging
import logging.config
import sys

import numpy as np
import yaml

import dataprovider
from config import ApplicationConfiguration


def handle_input(known):
    csv_structure = dataprovider.CsvCorpusStructure("../data")
    print(",".join("0" for _ in range(len(csv_structure.known) + len(csv_structure.unknown) - 2)))


def handle_all_inputs(config: ApplicationConfiguration):
    csv_structure = dataprovider.CsvCorpusStructure("../data")
    args = sys.argv[1:]
    if args:
        if len(args) != len(csv_structure.known):
            print("Warning! This program takes 0 or " + str(len(csv_structure.known)) + " parameters but not " + str(len(args)), file=sys.stderr)
        else:
            handle_input(args)
    corpus = dataprovider.CorpusProvider("../data", np.dtype(np.float16))
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
