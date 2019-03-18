#!/usr/bin/env python3
import logging
import logging.config
import sys
import csv
from io import TextIOBase

import tensorflow as tf
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
    corpus = dataprovider.CorpusProvider("../data", np.dtype(np.float16), False)
    known = corpus.known.shape
    unknown = corpus.unknown.shape
    unknown_ww = corpus.unknown_without_win.shape
    interesting = corpus.interesting.shape
    interesting_ww = corpus.interesting_without_win.shape
    win = corpus.win.shape
    print("{!s}\n\n{!s}\n\n{!s}\n\n{!s}\n\n{!s}\n\n{!s}".format(known, unknown, unknown_ww, interesting, interesting_ww, win))
    if config.should_read_stdin:
        input_reader = csv.DictReader(sys.stdin, fieldnames=csv_structure.known)
        for input_dict in input_reader:
            handle_input([input_dict[col] for col in known_columns])


if __name__ == '__main__':
    with open("logging.yaml", 'rt') as logging_yaml_file:
        logging.config.dictConfig(yaml.safe_load(logging_yaml_file.read()))
    handle_all_inputs(ApplicationConfiguration())
