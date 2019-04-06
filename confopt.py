# standard libraries
import pickle

import hyperopt
import logging

# specific libraries
from hyperopt import hp, Trials, fmin, tpe

# custom libraries
import tfnn
from config import *
from dataprovider import DataProvider


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
            BATCH_SIZE.key: hp.qloguniform(BATCH_SIZE.name, math.log(2 ** 9), math.log(2 ** 13), 1),
            ACTIVATION.key: hp.choice(ACTIVATION.name, (
                'relu',
                'leaky-relu',
                'sigmoid',
                'tanh',
            )),
        }

    @staticmethod
    def config_from_hyperopt_parameters(hyperopt_parameters) -> ApplicationConfiguration:
        config: ApplicationConfiguration = ApplicationConfiguration("")
        reversed_hls = list()
        base_node_amount = 0
        for width in hyperopt_parameters[HIDDEN_LAYERS.key]:
            reversed_hls.append(int(base_node_amount + width))
            base_node_amount += width
        config.set(HIDDEN_LAYERS, list(reversed(reversed_hls)))
        for config_option in (
            LEARNING_RATE,
            OPTIMIZER,
            BATCH_SIZE,
            SEED,
            ACTIVATION,
        ):
            value = hyperopt_parameters[config_option.key]
            config.set(config_option, str(value))
        return config

    def __init__(self, data_provider: DataProvider):
        self.logger = logging.getLogger('confopt')
        self.data_provider = data_provider

    def get_fmin_loss_from_hyperopt_parameters(self, hyperopt_parameters: Mapping[str, Any]):
        hyperparameters = ConfigurationOptimizer.config_from_hyperopt_parameters(hyperopt_parameters)
        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:
            loss = self.get_loss(hyperparameters)
        except InvalidArgumentError as e:
            message: str = e.message
            if 'Nan in summary histogram' in message:
                self.logger.warning("Error 'Nan in summary histogram' appeared with configuration {config!s}.".format(config=hyperopt_parameters))
                return {
                    'status': hyperopt.STATUS_FAIL
                }
            else:
                raise
        return {
            'status': hyperopt.STATUS_OK,
            'loss': loss,
        }

    def get_loss(self, config: ApplicationConfiguration) -> float:
        nn = tfnn.TrainableNeuralNetwork(self.data_provider, config)
        nn.train("../data/generator_state_{label:s}.npz".format(label=str(config)))
        return 1 - nn.get_best_seen_evaluation_accuracy()

    def optimize(self):
        trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
        while True:
            try:  # try to load an already saved trials object
                trials = pickle.load(open("hyper_parameter_loss_optimization_state.hyperopt", "rb"))
                self.logger.debug("Found saved Trials for hyperopt! Loading...")
                max_trials = len(trials.trials) + trials_step
                self.logger.debug("Rerunning from {:d} trials to {:d}".format(len(trials.trials), max_trials))
            except IOError:  # create a new trials object and start searching
                trials = Trials()
                max_trials = trials_step
            best = fmin(
                fn=self.get_fmin_loss_from_hyperopt_parameters,
                space=ConfigurationOptimizer.get_hyperopt_parameters(),
                algo=tpe.suggest,
                max_evals=max_trials,
                trials=trials,
            )
            self.logger.debug("Trial done. Current best score: {!s}".format(best))

            # save the trials object
            self.logger.debug("Saving trials...")
            with open("hyper_parameter_loss_optimization_state.hyperopt", "wb") as f:
                pickle.dump(trials, f)
            self.logger.debug("Saved trials.")
