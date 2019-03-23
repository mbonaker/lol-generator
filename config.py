from typing import *
import argparse
import numpy as np


class ApplicationConfiguration:
    @staticmethod
    def suffixed_bytes_to_bytes(suffixed_bytes: str) -> int:
        prepared_string = suffixed_bytes.lower().strip(" \t\nb")
        return ApplicationConfiguration.suffixed_si_to_number(prepared_string)

    @staticmethod
    def suffixed_si_to_number(suffixed_si: str) -> int:
        units = (
            ("k", 10 ** 3),
            ("ki", 1 << 10),
            ("m", 10 ** 6),
            ("mi", 1 << 20),
            ("g", 10 ** 9),
            ("gi", 1 << 30),
            ("t", 10 ** 12),
            ("ti", 1 << 40),
            ("p", 10 ** 15),
            ("pi", 1 << 50),
        )
        prepared_string = suffixed_si.strip().lower()
        try:
            return int(float(prepared_string))
        except ValueError:
            pass
        for suffix, factor in units:
            if prepared_string.endswith(suffix):
                return int(float(prepared_string[:-len(suffix)].rstrip()) * factor)
        raise ValueError("Can not convert {!r} to an integer.".format(suffixed_si))

    def __init__(self, str_arguments: Sequence[str]) -> None:
        argument_parser = argparse.ArgumentParser()
        argument_parser.add_argument(
            '-s', '--stdin',
            action='store_true',
            help="If this flag is set, input from stdin is used as 'known' data for which the 'unknown' data will be generated. Provide the data with one match per line (use the new line character alias \"\n\"). The data per match needs to be comma separated data columns in the order that can be seen in the file 'columns/known'.",
        )
        argument_parser.add_argument(
            '-t', '--train',
            action='store_true',
            help="If this flag is set, the corpus will be used to train the neural network.",
        )
        self.arguments = argument_parser.parse_args(str_arguments)

    @property
    def should_read_stdin(self) -> bool:
        return self.arguments.stdin

    @property
    def should_train(self) -> bool:
        return self.arguments.train

    @property
    def test_data_amount(self) -> int:
        return 1 << 14  # ~16 k

    @property
    def validation_data_amount(self) -> int:
        return 1 << 14  # ~16 k

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float16)

    @property
    def batch_size(self) -> int:
        return 1 << 13  # ~8 k

    @property
    def hidden_layer_structure(self) -> Tuple[int, ...]:
        return 2000, 500, 20

    @property
    def seed(self) -> int:
        return 0

    @property
    def lambda_(self) -> float:
        return 0

    @property
    def tensorboard_path(self) -> str:
        return "./tensorboard/5"

    @property
    def steps(self) -> int:
        return 10000

    @property
    def optimizer(self):
        import tensorflow as tf

        return tf.train.AdamOptimizer(
            learning_rate=0.01,
            epsilon=1e-4,
        )

    def __str__(self) -> str:
        return repr(self.arguments)
