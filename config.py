import logging
import math
import time
from typing import *
import argparse
import numpy as np


class ConfigurationOption:
    def __init__(self, key: str, name: str, is_determinant: bool, value_to_str: Callable[[Any], str], str_to_value: Callable[[str], Any]):
        self.key = key
        self.name = name
        self.is_determinant = is_determinant
        self.value_to_str = value_to_str
        self.str_to_value = str_to_value

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        return self.key


def ignored_columns_to_str(ignored_columns: Iterable[str]) -> str:
    keywords = set()
    for col in ignored_columns:
        if ".bans." in col:
            keywords.add('bans')
            assert all("teams.{:d}.bans.{:d}.championId".format(i, j) in ignored_columns for i in (0, 1) for j in (0, 1, 2, 3, 4))
        else:
            keywords.update(col)
    if not keywords:
        return "none"
    else:
        return ":".join(keywords)


def str_to_ignored_columns(keyword_str: str) -> Iterable[str]:
    if not keyword_str or keyword_str == "none":
        return []
    keywords = keyword_str.split(":")
    cols = set()
    for keyword in keywords:
        if keyword == "bans":
            cols.update("teams.{:d}.bans.{:d}.championId".format(i, j) for i in (0, 1) for j in (0, 1, 2, 3, 4))
        else:
            cols.add(keyword)
    return cols


def optimize_adam(lr):
    import tensorflow as tf
    return tf.train.AdamOptimizer(lr, epsilon=1e-4)


def optimize_adagrad(lr):
    import tensorflow as tf
    return tf.train.AdagradOptimizer(lr)


def optimize_adadelta(lr):
    import tensorflow as tf
    return tf.train.AdadeltaOptimizer(lr, epsilon=1e-4)


def optimize_sgd(lr):
    import tensorflow as tf
    return tf.train.GradientDescentOptimizer(lr)


def optimize_momentum(lr):
    import tensorflow as tf
    return tf.train.MomentumOptimizer(lr, 0.1)


def str_to_optimizer(keyword: str) -> Callable[[str], Any]:
    if keyword == 'adam':
        return optimize_adam
    elif keyword == 'adagrad':
        return optimize_adagrad
    elif keyword == 'adadelta':
        return optimize_adadelta
    elif keyword == 'sgd':
        return optimize_sgd
    elif keyword == 'momentum':
        return optimize_momentum
    else:
        raise ValueError("Keyword {!r} not understood".format(keyword))


def optimizer_to_str(optimizer) -> str:
    instance = optimizer(0.1)
    classname = type(instance).__name__
    if classname.endswith('Optimizer'):
        return classname[:-9].lower()
    else:
        return classname


def activation_function_to_str(fun: Callable) -> str:
    import tensorflow as tf

    return {
        tf.nn.leaky_relu: 'leaky-relu',
        tf.nn.relu: 'relu',
        tf.nn.sigmoid: 'sigmoid',
        tf.nn.tanh: 'tanh',
    }[fun]


def str_to_activation_function(keyword: str) -> Callable:
    import tensorflow as tf

    return {
        'leaky-relu': tf.nn.leaky_relu,
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh,
    }[keyword]


def suffixed_si_to_number(suffixed_si: str) -> int:
    units = (
        ("K", 10 ** 3),
        ("Ki", 1 << 10),
        ("M", 10 ** 6),
        ("Mi", 1 << 20),
        ("G", 10 ** 9),
        ("Gi", 1 << 30),
        ("T", 10 ** 12),
        ("Ti", 1 << 40),
        ("P", 10 ** 15),
        ("Pi", 1 << 50),
    )
    prepared_string = suffixed_si.strip()
    try:
        return int(float(prepared_string))
    except ValueError:
        pass
    for suffix, factor in units:
        if prepared_string.endswith(suffix):
            return int(float(prepared_string[:-len(suffix)].rstrip()) * factor)
    raise ValueError("Can not convert {!r} to an integer.".format(suffixed_si))


def suffixed_bytes_to_bytes(suffixed_bytes: str) -> int:
    prepared_string = suffixed_bytes.lower().strip(" \t\nb")
    return suffixed_si_to_number(prepared_string)


class TrainingStatus:
    def __init__(self, config, done_batches: int, loss: float):
        self.config = config
        self.done_batches = done_batches
        self.loss = loss


class StopCriterion:
    def update(self, status: TrainingStatus):
        raise NotImplementedError()

    def should_stop(self):
        raise NotImplementedError()


class MaxTrainingCriterion(StopCriterion):
    def __init__(self, training_sample_amount: int):
        self.max = training_sample_amount
        self.current = None
        self.logger = logging.getLogger(__name__)

    def update(self, status: TrainingStatus):
        self.current = status.done_batches * status.config.batch_size

    def should_stop(self):
        if self.current is not None and self.current >= self.max:
            self.logger.log(logging.INFO, "Signaled stopping due max training")
            return True
        else:
            return False


class StagnationCriterion(StopCriterion):
    def __init__(self, min_log: int, checkpoints: Iterable[float]):
        self.min_log = min_log
        self.checkpoints = checkpoints
        self.loss_log = []
        self.logger = logging.getLogger(__name__)

    def update(self, status: TrainingStatus):
        self.loss_log.append(status.loss)

    def should_stop(self):
        if len(self.loss_log) < self.min_log:
            return False
        for checkpoint in self.checkpoints:
            if self.loss_log[math.floor(checkpoint * (len(self.loss_log) - 1))] >= self.loss_log[-1]:
                return False
        self.logger.log(logging.INFO, "Signaled stopping due to stagnation")
        return True


class TimingCriterion(StopCriterion):
    def __init__(self, max_seconds: float):
        self.max = max_seconds
        self.start = None
        self.logger = logging.getLogger(__name__)

    def update(self, status: TrainingStatus):
        if self.start is None:
            self.start = time.perf_counter()

    def should_stop(self):
        if self.start is None:
            return False
        if self.start + self.max <= time.perf_counter():
            self.logger.log(logging.INFO, "Signaled stopping due to timing")
            return True
        else:
            return False


class StopCriteria(StopCriterion):
    def __init__(self, keywords: str):
        self.logger = logging.getLogger(__name__)
        self.criteria = []
        if keywords and keywords != 'never':
            for keyword in keywords.split(":"):
                if keyword.startswith('samples'):
                    self.criteria.append(MaxTrainingCriterion(suffixed_si_to_number(keyword[7:])))
                elif keyword.startswith('stagnant'):
                    args = keyword[8:].split(',')
                    min_log = 30 if len(args) == 0 or args[0] == '' else int(args[0])
                    checkpoints = [0.75, 0.8] if len(args) < 2 else [float(arg) for arg in args[1:]]
                    self.criteria.append(StagnationCriterion(min_log, checkpoints))
                elif keyword.startswith('seconds'):
                    self.criteria.append(TimingCriterion(suffixed_si_to_number(keyword[7:])))
                else:
                    NotImplementedError()

    def update(self, status: TrainingStatus):
        for criterion in self.criteria:
            criterion.update(status)

    def should_stop(self):
        for criterion in self.criteria:
            if criterion.should_stop():
                self.logger.log(logging.DEBUG, "Sub-Criterion signaled stopping for {!s}".format(self))
                return True
        return False

    def __str__(self):
        keywords = []
        for criterion in self.criteria:
            if isinstance(criterion, MaxTrainingCriterion):
                keywords.append("samples{:d}".format(criterion.max))
            elif isinstance(criterion, StagnationCriterion):
                c = ",".join("{:g}".format(checkpoint) for checkpoint in criterion.checkpoints)
                keywords.append("stagnant{min:d},{checkpoints:s}".format(min=criterion.min_log, checkpoints=c))
            elif isinstance(criterion, TimingCriterion):
                keywords.append("seconds{:g}".format(criterion.max))
            else:
                raise NotImplementedError()
        return ":".join(keywords)

    def __eq__(self, other):
        return str(self) == str(other)


LEARNING_RATE = ConfigurationOption('lr', 'Learning Rate', True, lambda x: "{:.0e}".format(x), float)
LABEL = ConfigurationOption('label', 'Label', False, lambda x: 'none' if x is None else x, str)
DTYPE = ConfigurationOption('dt', 'DType', True, lambda x: x.str, np.dtype)
OPTIMIZER = ConfigurationOption('opti', 'Optimizer', True, optimizer_to_str, str_to_optimizer)
BATCH_SIZE = ConfigurationOption('bs', 'Batch Size', True, lambda x: "{:.1e}".format(x), suffixed_si_to_number)
SEED = ConfigurationOption('seed', 'Seed', True, str, int)
IGNORED_COLUMNS = ConfigurationOption('ic', 'Ignored Columns', False, ignored_columns_to_str, str_to_ignored_columns)
HIDDEN_LAYERS = ConfigurationOption('hl', 'Hidden Layer Structure', True, lambda x: ":".join(str(n) for n in x), lambda x: list(int(n) for n in x.split(":")))
ACTIVATION = ConfigurationOption('af', 'Activation Function', True, activation_function_to_str, str_to_activation_function)
TEST_DATA_AMOUNT = ConfigurationOption('td', 'Test Data Amount', True, str, suffixed_si_to_number)
VALIDATION_DATA_AMOUNT = ConfigurationOption('vd', 'Validation Data Amount', True, str, suffixed_si_to_number)
LAMBDA = ConfigurationOption('lambda', 'Lambda for L2-Regularization', True, str, float)
STOP_CRITERIA = ConfigurationOption('stop', 'Stop Criteria', True, str, StopCriteria)
SAMPLES_PER_TEST_EVALUATION = ConfigurationOption('evtest', 'Samples per Test Evaluation', True, str, suffixed_si_to_number)
SAMPLES_PER_TRAIN_EVALUATION = ConfigurationOption('evtrain', 'Samples per Train Evaluation', True, str, suffixed_si_to_number)
CODE_VERSION = ConfigurationOption('v', 'Code Version', True, str, int)

OPTIONS = (
    LEARNING_RATE,
    LABEL,
    DTYPE,
    OPTIMIZER,
    BATCH_SIZE,
    SEED,
    IGNORED_COLUMNS,
    HIDDEN_LAYERS,
    ACTIVATION,
    TEST_DATA_AMOUNT,
    VALIDATION_DATA_AMOUNT,
    LAMBDA,
    STOP_CRITERIA,
    SAMPLES_PER_TEST_EVALUATION,
    SAMPLES_PER_TRAIN_EVALUATION,
    CODE_VERSION,
)


class ApplicationConfiguration:

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
        argument_parser.add_argument(
            '-o', '--hyperopt',
            action='store_true',
            help="If this flag is set, the hyper parameter optimization is done.",
        )
        argument_parser.add_argument(
            '-l', '--label',
            type=LABEL.str_to_value,
            help="The name of this instance running (used for tensorboard).",
        )
        argument_parser.add_argument(
            '--ic',
            type=IGNORED_COLUMNS.str_to_value,
            help="Columns to ignore",
        )
        argument_parser.add_argument(
            '--hl',
            type=HIDDEN_LAYERS.str_to_value,
            help=HIDDEN_LAYERS.name,
        )
        argument_parser.add_argument(
            '--stop',
            type=STOP_CRITERIA.str_to_value,
            help=STOP_CRITERIA.name,
        )
        argument_parser.add_argument(
            '--bs',
            type=BATCH_SIZE.str_to_value,
            help=BATCH_SIZE.name,
        )
        argument_parser.add_argument(
            '--opti', '--optim',
            type=OPTIMIZER.str_to_value,
            help=OPTIMIZER.name,
        )
        argument_parser.add_argument(
            '--evtest',
            type=SAMPLES_PER_TEST_EVALUATION.str_to_value,
            help=SAMPLES_PER_TEST_EVALUATION.name,
        )
        argument_parser.add_argument(
            '--evtrain',
            type=SAMPLES_PER_TRAIN_EVALUATION.str_to_value,
            help=SAMPLES_PER_TRAIN_EVALUATION.name,
        )
        argument_parser.add_argument(
            '-λ', '--lambda',
            type=LAMBDA.str_to_value,
            help=LAMBDA.name,
        )
        argument_parser.add_argument(
            '--lr',
            type=LEARNING_RATE.str_to_value,
            help=LEARNING_RATE.name,
        )
        self.arguments = argument_parser.parse_args(str_arguments)
        self.default_options = {
            LAMBDA: 0,
            DTYPE: 'float16',
            HIDDEN_LAYERS: [256],
            LEARNING_RATE: 0.001,
            OPTIMIZER: 'adam',
            ACTIVATION: 'leaky-relu',
            BATCH_SIZE: 2048,
            SEED: 0,
            TEST_DATA_AMOUNT: 1 << 14,
            VALIDATION_DATA_AMOUNT: 1 << 14,
            STOP_CRITERIA: StopCriteria('seconds1800:stagnant'),  # 1800 seconds is half an hour
            IGNORED_COLUMNS: '',
            SAMPLES_PER_TEST_EVALUATION: 500000,
            SAMPLES_PER_TRAIN_EVALUATION: 100000,
            CODE_VERSION: 12,
        }
        self.option_dict = {}

    @property
    def shoud_optimize_hyperparameters(self) -> bool:
        return self.arguments.hyperopt

    @property
    def should_read_stdin(self) -> bool:
        return self.arguments.stdin

    @property
    def should_train(self) -> bool:
        return self.arguments.train

    @property
    def test_data_amount(self) -> int:
        return self.get_value(TEST_DATA_AMOUNT)

    @property
    def validation_data_amount(self) -> int:
        return self.get_value(VALIDATION_DATA_AMOUNT)

    @property
    def dtype(self) -> np.dtype:
        return self.get_value(DTYPE)

    @property
    def batch_size(self) -> int:
        return self.get_value(BATCH_SIZE)

    @property
    def hidden_layer_structure(self) -> Tuple[int, ...]:
        return self.get_value(HIDDEN_LAYERS)

    @property
    def seed(self) -> int:
        return self.get_value(SEED)

    @property
    def ignored_columns(self) -> Iterable[str]:
        return self.get_value(IGNORED_COLUMNS)

    @property
    def lambda_(self) -> float:
        return self.get_value(LAMBDA)

    @property
    def label(self) -> str:
        return self.get_value(LABEL)

    @property
    def stop_criterion(self) -> StopCriterion:
        return self.get_value(STOP_CRITERIA)

    @property
    def tensorboard_path(self) -> str:
        return "./tensorboard/{label}".format(label=str(self))

    @property
    def optimizer(self) -> Callable[[float], Any]:
        return self.get_value(OPTIMIZER)

    @property
    def samples_per_test_evaluation(self) -> int:
        return self.get_value(SAMPLES_PER_TEST_EVALUATION)

    @property
    def samples_per_train_evaluation(self) -> int:
        return self.get_value(SAMPLES_PER_TRAIN_EVALUATION)

    @property
    def learning_rate(self) -> float:
        return self.get_value(LEARNING_RATE)

    @property
    def activation_function(self):
        return self.get_value(ACTIVATION)

    def set(self, option: ConfigurationOption, value):
        self.option_dict[option] = value

    def get_value(self, option: ConfigurationOption):
        if option in self.option_dict:
            value = self.option_dict[option]
            if type(value) == str:
                return option.str_to_value(value)
            else:
                return value
        if hasattr(self.arguments, option.key) and getattr(self.arguments, option.key) is not None:
            return getattr(self.arguments, option.key)
        if option in self.default_options:
            return self.get_default(option)
        return None

    def get_default(self, option: ConfigurationOption):
        value = self.default_options[option]
        if type(value) == str:
            return option.str_to_value(value)
        else:
            return value

    def __str__(self) -> str:
        information_points = []
        for option in OPTIONS:
            if option not in self.option_dict and not hasattr(self.arguments, option.key):
                continue
            value = self.get_value(option)
            if value is None or option in self.default_options and self.get_default(option) == value:
                continue
            info = "{:s}={:s}".format(option.key, option.value_to_str(self.get_value(option)))
            information_points.append(info)
        if not information_points:
            return "default-v{:d}".format(self.get_value(CODE_VERSION))
        else:
            return "v{v:d}_{info:s}".format(v=self.get_value(CODE_VERSION), info="_".join(information_points))
