#!/usr/bin/env python3
import logging
import logging.config
import sys
import csv
import yaml


def handle_input(known):
    with open('columns/unknown', 'r') as column_file:
        unknown_columns = list(column_file)
    print(",".join("0" for _ in range(len(known) + len(unknown_columns) - 1)))


def handle_all_inputs():
    with open('columns/known', 'r') as column_file:
        known_columns = list(column_file)
    args = sys.argv[1:]
    if args:
        if len(args) != len(known_columns):
            print("Warning! This program takes 0 or " + str(len(known_columns)) + " parameters but not " + str(len(args)), file=sys.stderr)
        else:
            handle_input(args)
    if not sys.stdin.isatty():
        input_reader = csv.DictReader(sys.stdin, fieldnames=known_columns)
        for input_dict in input_reader:
            handle_input([input_dict[col] for col in known_columns])


if __name__ == '__main__':
    with open("logging.yaml", 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    handle_all_inputs()
