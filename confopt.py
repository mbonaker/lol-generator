# standard libraries
import math

# specific libraries
from hyperopt import hp

# custom libraries
from config import *


class ConfigurationOptimizer:

    @staticmethod
    def get_hyperopt_parameters():
        return {
            LEARNING_RATE.key:
                hp.loguniform(LEARNING_RATE.name, math.log(0.000001), math.log(0.1)),
            HIDDEN_LAYERS.key:
                hp.choice(HIDDEN_LAYERS.name, (
                    [
                        hp.qloguniform('layer_1_1', math.log(1), math.log(2 ** 10), 1),
                    ], [
                        hp.qloguniform('layer_2_2', math.log(1), math.log(2 ** 10), 1),
                        hp.qloguniform('layer_2_1', math.log(1), math.log(2 ** 10), 1) - 1,
                    ], [
                        hp.qloguniform('layer_3_3', math.log(1), math.log(2 ** 10), 1),
                        hp.qloguniform('layer_3_2', math.log(1), math.log(2 ** 10), 1) - 1,
                        hp.qloguniform('layer_3_1', math.log(1), math.log(2 ** 10), 1) - 1,
                    ], [
                        hp.qloguniform('layer_4_4', math.log(1), math.log(2 ** 10), 1),
                        hp.qloguniform('layer_4_3', math.log(1), math.log(2 ** 10), 1) - 1,
                        hp.qloguniform('layer_4_2', math.log(1), math.log(2 ** 10), 1) - 1,
                        hp.qloguniform('layer_4_1', math.log(1), math.log(2 ** 10), 1) - 1,
                    ], [
                        hp.qloguniform('layer_5_5', math.log(1), math.log(2 ** 10), 1),
                        hp.qloguniform('layer_5_4', math.log(1), math.log(2 ** 10), 1) - 1,
                        hp.qloguniform('layer_5_3', math.log(1), math.log(2 ** 10), 1) - 1,
                        hp.qloguniform('layer_5_2', math.log(1), math.log(2 ** 10), 1) - 1,
                        hp.qloguniform('layer_5_1', math.log(1), math.log(2 ** 10), 1) - 1,
                    ],
                )),
            OPTIMIZER.key:
                hp.choice(OPTIMIZER.name, [
                    'adam',
                    'adagrad',
                    'momentum',
                    'adadelta',
                    'sgd',
                ]),
            SEED.key: 0,
            DTYPE.key: hp.choice(DTYPE.name, ('float16', 'float32')),
            BATCH_SIZE.key: hp.qloguniform(BATCH_SIZE.name, math.log(2 ** 9), math.log(2 ** 15), 1),
            ACTIVATION.key: hp.choice(ACTIVATION.name, (
                'relu',
                'leaky_relu',
                'sigmoid',
                'tanh',
            )),
        }

    @staticmethod
    def from_hyperopt_parameters(hyperopt_parameters, config: ApplicationConfiguration):
        reversed_hls = list()
        base_node_amount = 0
        for width in hyperopt_parameters[HIDDEN_LAYERS.key]:
            reversed_hls.append(int(base_node_amount + width))
            base_node_amount += width
        config.set(list(reversed(reversed_hls)), HIDDEN_LAYERS)
        for config_option in (
            LEARNING_RATE,
            OPTIMIZER,
            BATCH_SIZE,
            SEED,
            IGNORED_COLUMNS,
            ACTIVATION,
        ):
            value = hyperopt_parameters[config_option.key]
            config.set(value, config_option)
