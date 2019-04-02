import logging
import io
from collections import OrderedDict
from time import perf_counter
from typing import Optional, TextIO

import tensorflow as tf
import numpy as np

import dataprovider as dp
from config import ApplicationConfiguration
from config import TrainingStatus
from nn import NeuralNetwork

PURPOSE_TRAIN = 1
PURPOSE_TEST = 2
PURPOSE_PREDICT = 4


class TrainableNeuralNetwork(NeuralNetwork):

    def __init__(self, data: dp.DataProvider, config: ApplicationConfiguration):
        super().__init__(data, config)

        unknown_csv_structure = dp.CsvCorpusStructure(self.data.data_path, dp.PORTION_UNKNOWN - dp.PORTION_WIN, True)
        unknown_data_structure = dp.NumpyCorpusStructure(unknown_csv_structure, self.config.dtype, dp.PORTION_UNKNOWN - dp.PORTION_WIN, True)

        self.random_state = np.random.RandomState(config.seed)
        #
        # Create the dataset and iterator
        #
        train_data_shape = (
            (None, data.known.shape[1]),
            (None, data.unknown_without_win.shape[1]),
        )
        train_data = tf.data.Dataset.from_generator(
            self.make_train_ndarray_batch_generator(),
            output_types=(config.dtype, config.dtype),
            output_shapes=train_data_shape,
        )
        self.logger.log(logging.DEBUG, "Repeat train dataset")
        train_data = train_data.repeat()
        self.logger.log(logging.DEBUG, "Prefetch train dataset")
        train_data = train_data.prefetch(1)
        self.logger.log(logging.DEBUG, "Unbatch train dataset")
        train_data = train_data.apply(tf.data.experimental.unbatch())
        self.logger.log(logging.DEBUG, "Shuffle train dataset")
        train_data = train_data.shuffle(self.config.batch_size)
        self.logger.log(logging.DEBUG, "Batch train dataset")
        train_data = train_data.batch(self.config.batch_size, drop_remainder=False)
        test_data_shape = (
            (None, data.known.shape[1]),
            (None, data.unknown_without_win.shape[1]),
        )
        self.logger.log(logging.DEBUG, "Create test dataset with shape {!r}".format(test_data_shape))
        test_data = tf.data.Dataset.from_generator(
            self.make_test_ndarray_generator(),
            output_types=(config.dtype, config.dtype),
            output_shapes=test_data_shape,
        )
        self.logger.log(logging.DEBUG, "Repeat test dataset")
        test_data = test_data.repeat()
        self.train_iterator = train_data.make_initializable_iterator()
        self.test_iterator = test_data.make_initializable_iterator()
        self.iterator_handle = tf.placeholder(tf.string, shape=[])
        data_iterator = tf.data.Iterator.from_string_handle(self.iterator_handle, train_data.output_types, train_data.output_shapes)

        #
        # Create the structure of the DNN (hidden layers and stuff)
        #
        self.logger.log(logging.DEBUG, "Create NN logic graph")
        previous_layer_size = data.known.shape[1]
        target_amount = data.unknown_without_win.shape[1]
        (self.features, self.targets) = data_iterator.get_next(name="layer_in")
        x = tf.cast(tf.transpose(self.features), config.dtype)
        x_is_sparse = True
        y = self.targets
        self.layers = [(None, None, self.features)]
        for i, l in enumerate(config.hidden_layer_structure):
            self.logger.log(logging.DEBUG, "Add layer {i:d} with {l:d} nodes to the computation".format(i=i, l=l))
            w = tf.Variable(tf.random_normal((previous_layer_size, l), 0, 0.05, dtype=config.dtype, seed=config.seed), name="weights_" + str(i))
            b = tf.Variable(tf.random_normal((l, 1), 0, 0.05, dtype=config.dtype, seed=config.seed), name="bias_" + str(i))
            sum_w = tf.matmul(w, x, True, b_is_sparse=x_is_sparse)
            x_is_sparse = False
            x = config.activation_function(tf.add(sum_w, b), name="layer_" + str(i))
            self.layers.append((w, b, x))
            previous_layer_size = l
        w = tf.Variable(tf.random_normal((previous_layer_size, target_amount), 0, 0.05, dtype=config.dtype, seed=config.seed), name="weights_end")
        b = tf.Variable(tf.random_normal((target_amount, 1), 0, 0.05, dtype=config.dtype, seed=config.seed), name="bias_end")
        logit = tf.transpose(tf.add(tf.matmul(w, x, True), b), name="logit")

        #
        # Calculate the loss
        #
        self.losses = OrderedDict()
        self.labels = OrderedDict()
        self.accuracies = OrderedDict()
        self.accuracy_amount = 0
        self.logger.log(logging.DEBUG, "Add loss calculation to the graph...")

        def ignore_col(col: dp.CsvColumnSpecification) -> bool:
            return col.name in self.config.ignored_columns

        for data_slice in unknown_data_structure.generate_slices(lambda c: not ignore_col(c) and c.handling == dp.CsvColumnSpecification.HANDLING_CONTINUOUS):
            ys = y[:, data_slice]
            logits = logit[:, data_slice]
            loss = tf.reduce_mean(tf.reduce_sum(tf.abs(ys - logits), axis=1))
            self.losses['continuous_{!r}'.format(data_slice)] = loss
            self.accuracies['continuous_{!r}'.format(data_slice)] = tf.reduce_sum((tf.abs(ys - logits) + 1) ** -1, axis=1)
            self.accuracy_amount += tf.shape(logits)[1]
        for csv_column in unknown_csv_structure.columns:
            data_slice = unknown_data_structure.csv_column_spec_to_np_slice(csv_column)
            ys = y[:, data_slice]
            logits = logit[:, data_slice]
            handling = csv_column.handling if csv_column.name not in config.ignored_columns else dp.CsvColumnSpecification.HANDLING_NONE
            if handling == dp.CsvColumnSpecification.HANDLING_BOOL:
                loss = tf.cast(tf.losses.hinge_loss(labels=ys, logits=logits, reduction=tf.losses.Reduction.MEAN), self.config.dtype)
                self.losses['bool_{:s}'.format(csv_column.name)] = loss
                self.labels['bool_{:s}'.format(csv_column.name)] = ys
                self.accuracies['bool_{:s}'.format(csv_column.name)] = tf.reduce_sum(tf.cast(tf.equal(logits > 0, ys > 0), self.config.dtype), axis=1)
                self.accuracy_amount += tf.shape(logits)[1]
        # per participant one-hot encodings
        for pid in range(0, 9):
            for key_part in ("highestAchievedSeasonTier", "timeline.lane", "timeline.role"):
                key = 'participants.{id:d}.{k:s}'.format(id=pid, k=key_part)
                slice_ = unknown_data_structure.csv_column_name_to_np_slice(key)
                logits = logit[:, slice_]
                ys = y[:, slice_]
                loss = tf.cast(tf.losses.hinge_loss(labels=ys, logits=logits, reduction=tf.losses.Reduction.MEAN), self.config.dtype)
                self.losses['onehot_{:s}'.format(key)] = loss
                self.labels['onehot_{:s}'.format(key)] = ys
                self.accuracies['onehot_{:s}'.format(key)] = tf.cast(tf.equal(tf.argmax(logits, 1, output_type=tf.int32), tf.argmax(ys, 1, output_type=tf.int32)), self.config.dtype)
                self.accuracy_amount += 1
        for team_id in (0, 1):
            ban_slices = []
            for ban_id in range(0, 4):
                ban_name = "teams.{tid:d}.bans.{bid:d}.championId".format(tid=team_id, bid=ban_id)
                ban_slice = unknown_data_structure.csv_column_name_to_np_slice(ban_name)
                ban_slices.append(ban_slice)
                logits = logit[:, ban_slice]
                ys = y[:, ban_slice]
                self.accuracies["onehot_teams.{tid:d}.bans.X.championId".format(tid=team_id)] = tf.cast(tf.equal(tf.argmax(logits, 1, output_type=tf.int32), tf.argmax(ys, 1, output_type=tf.int32)), self.config.dtype)
                self.accuracy_amount += 1
            logits = sum(logit[:, ban_slice] for ban_slice in ban_slices)
            ys = sum(y[:, ban_slice] for ban_slice in ban_slices)
            loss = tf.cast(tf.losses.hinge_loss(labels=ys, logits=logits, reduction=tf.losses.Reduction.MEAN), self.config.dtype)
            self.losses["onehot_teams.{tid:d}.bans.X.championId".format(tid=team_id)] = loss
            self.labels["onehot_teams.{tid:d}.bans.X.championId".format(tid=team_id)] = ys
        regularization = sum(tf.nn.l2_loss(w) for w, _, _ in self.layers if w is not None) * config.lambda_
        self.losses["regularization"] = regularization
        self.loss = tf.math.add_n(list(self.losses.values()), "loss")
        self.accuracy = tf.math.divide(tf.math.add_n(list(self.accuracies.values())), tf.cast(self.accuracy_amount, tf.float16), name="accuracy")

        #
        # Calculate the predictions as last layer
        #
        self.logger.log(logging.DEBUG, "Add prediction calculation to the graph...")
        prediction_slices = []
        for data_slice, handling in unknown_data_structure.generate_handling_slices(self.config.ignored_columns):
            logit_slice = logit[:, data_slice]
            if handling == dp.CsvColumnSpecification.HANDLING_NONE:
                y_hat_slice = tf.zeros_like(logit_slice)
            elif handling == dp.CsvColumnSpecification.HANDLING_ONEHOT or handling == dp.CsvColumnSpecification.HANDLING_BOOL:
                y_hat_slice = tf.nn.sigmoid(logit_slice)
            elif handling == dp.CsvColumnSpecification.HANDLING_CONTINUOUS:
                y_hat_slice = logit_slice
            else:
                raise NotImplementedError()
            prediction_slices.append(y_hat_slice)
        predictions = tf.concat(prediction_slices, axis=1)
        self.layers.append((w, b, predictions))

        #
        # Create the evaluation of the DNN (loss/error and stuff)
        #
        self.logger.log(logging.DEBUG, "Add summary calculation to the graph...")
        error = tf.subtract(predictions, y, "error")
        self.mae = tf.reduce_mean(tf.math.abs(error), axis=0, name="MAE")
        self.train_summaries = tf.summary.merge([
            # tf.summary.histogram("train_mean_error", self.mae),
            tf.summary.scalar("train_accuracy", tf.reduce_mean(self.accuracy)),
        ])
        test_evaluation_summaries = [
            tf.summary.scalar("test_accuracy", tf.reduce_mean(self.accuracy)),
        ]
        for csv_column in unknown_csv_structure.columns:
            column_slice = unknown_data_structure.csv_column_spec_to_np_slice(csv_column)
            column_mae = tf.reduce_mean(tf.abs(error[:, column_slice]))
            error_summary = tf.summary.scalar("test_error_{column_name:s}".format(column_name=csv_column.name), column_mae)
            test_evaluation_summaries.append(error_summary)
            column_prediction = predictions[:, column_slice]
            prediction_summary = tf.summary.histogram("test_prediction_{column_name:s}".format(column_name=csv_column.name), tf.reduce_mean(column_prediction, axis=1))
            test_evaluation_summaries.append(prediction_summary)
        self.test_summaries = tf.summary.merge(test_evaluation_summaries)


        #
        # Create the optimization of the DNN (gradient and stuff)
        #
        self.logger.log(logging.DEBUG, "Add optimization calculation to the graph...")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # grads_and_vars is a list of tuples (gradient, variable)
        parameters = [w for w, _, _ in self.layers if w is not None] + [b for _, b, _ in self.layers if b is not None]
        grads_and_vars = config.optimizer(config.learning_rate).compute_gradients(self.loss, var_list=parameters)
        capped_grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
        # apply the capped gradients.
        self.minimizer = config.optimizer(config.learning_rate).apply_gradients(capped_grads_and_vars, global_step=self.global_step)

        self.train_handle = None
        self.test_handle = None
        self.tf_writer: Optional[tf.summary.FileWriter] = None
        self.running_duration = 0
        self.need_tf_weight_reassignment = False

    def make_train_ndarray_batch_generator(self):
        def generate_ndarray_batches():
            ndarray = self.data.get_ndarray()[0:-self.config.test_data_amount - self.config.validation_data_amount, :]
            duration = self.data.np_structure.csv_column_name_to_np_slice("gameDuration")
            for batch in np.split(ndarray, np.arange(self.config.batch_size, ndarray.shape[0], self.config.batch_size), axis=0):
                # filter out 'remakes' (games that have been dissolved early due to leaving participants)
                keep = batch[:, duration] > (15 * 60 - 1762.503) / 493.163
                keep = keep[:, 0]
                x: np.ndarray = batch[keep, self.data.np_structure.known_slice]
                y: np.ndarray  = batch[keep, self.data.np_structure.unknown_without_win_slice]
                # take some optional information from x out (simulate incomplete user input)
                if isinstance(x, np.memmap) or not x.flags.writeable:
                    x_new = np.ndarray(x.shape, x.dtype)
                    x_new[:, :] = x
                    x = x_new
                self.data.np_structure.randomly_unspecify_optional_columns(x, self.random_state, 0.1)
                yield (x, y)
        return generate_ndarray_batches

    def make_test_ndarray_generator(self):
        def generate_ndarray():
            ndarray = self.data.get_ndarray()[-self.config.test_data_amount - self.config.validation_data_amount:-self.config.test_data_amount, :]
            # filter out 'remakes' (games that have been dissolved early due to leaving participants)
            duration = self.data.np_structure.csv_column_name_to_np_slice("gameDuration")
            keep = ndarray[:, duration] > (15 * 60 - 1762.503) / 493.163
            keep = keep[:, 0]
            x = ndarray[keep, self.data.np_structure.known_slice]
            y = ndarray[keep, self.data.np_structure.unknown_without_win_slice]
            # take some optional information from x out (simulate incomplete user input)
            if isinstance(x, np.memmap) or not x.flags.writeable:
                x_new = np.ndarray(x.shape, x.dtype)
                x_new[:, :] = x
                x = x_new
            self.data.np_structure.randomly_unspecify_optional_columns(x, self.random_state, 0.1)
            yield (x, y)
        return generate_ndarray

    def start_session(self, sess: tf.Session):
        self.logger.log(logging.DEBUG, "Initialize the iterators")
        self.train_handle = sess.run(self.train_iterator.string_handle())
        self.test_handle = sess.run(self.test_iterator.string_handle())
        sess.run(self.train_iterator.initializer)
        sess.run(self.test_iterator.initializer)
        sess.run(tf.global_variables_initializer())

        self.logger.log(logging.DEBUG, "Start the file writer for tensorboard")
        self.tf_writer = tf.summary.FileWriter(self.config.tensorboard_path, sess.graph, flush_secs=10)

    def train_one_step(self, sess: tf.Session):
        self.reassign_tf_weights_if_necessary(sess)
        time_before = perf_counter()
        sess.run(self.minimizer, feed_dict={self.iterator_handle: self.train_handle})
        self.running_duration += perf_counter() - time_before

    def train_eval(self, sess: tf.Session):
        self.reassign_tf_weights_if_necessary(sess)
        step, summaries, accuracy_amount, *accuracies = sess.run([
            self.global_step,
            self.train_summaries,
            self.accuracy_amount,
        ] + list(self.accuracies.values()), feed_dict={self.iterator_handle: self.train_handle})
        training_amount = step * self.config.batch_size
        self.tf_writer.add_summary(summaries, training_amount)

    def test_eval(self, sess: tf.Session) -> float:
        self.reassign_tf_weights_if_necessary(sess)
        step, summaries, loss = sess.run([
            self.global_step,
            self.test_summaries,
            self.loss,
        ], feed_dict={self.iterator_handle: self.test_handle})
        training_amount = step * self.config.batch_size
        self.tf_writer.add_summary(summaries, training_amount)
        return loss

    def stop_session(self):
        self.tf_writer.close()
        # tf.reset_default_graph()

    def reassign_tf_weights_if_necessary(self, sess: tf.Session):
        if self.need_tf_weight_reassignment:
            self.logger.log(logging.DEBUG, "Reassign the tensorflow weights.")
            for (np_w, np_b), (tf_w, tf_b, _) in zip(zip(self.weights, self.biases), (layer for layer in self.layers if layer[0] is not None)):
                sess.run(tf.assign(tf_w, np_w))
                sess.run(tf.assign(tf_b, np_b))

    def update_tf_weights_to_np_weights(self, sess: tf.Session):
        self.weights = sess.run([w for w, b, p in self.layers if w is not None])
        self.biases = sess.run([b for w, b, p in self.layers if b is not None])
        assert len(self.weights) == len(self.biases)

    def load_weights(self, file_name: str):
        super().load_weights(file_name)
        self.need_tf_weight_reassignment = True

    def save_tf_weights(self, file_name: str, sess: tf.Session):
        self.update_tf_weights_to_np_weights(sess)
        self.save_weights(file_name)

    def train(self, file_name: str):
        smallest_loss = float('inf')
        with tf.Session() as session:
            self.start_session(session)
            trained_batches = 0
            samples_since_test_evaluation = 0
            samples_since_train_evaluation = 0
            self.logger.log(logging.DEBUG, "Start training loop")
            while True:
                samples = trained_batches * self.config.batch_size
                samples_since_test_evaluation += self.config.batch_size
                samples_since_train_evaluation += self.config.batch_size
                if samples == 0 or samples_since_train_evaluation >= self.config.samples_per_train_evaluation:
                    self.logger.log(logging.DEBUG, "Evaluate the network on training data...")
                    self.train_eval(session)
                    samples_since_train_evaluation = 0
                if samples == 0 or samples_since_test_evaluation >= self.config.samples_per_test_evaluation:
                    self.logger.log(logging.DEBUG, "Evaluate the network on test data...")
                    loss = self.test_eval(session)
                    samples_since_test_evaluation = 0
                    if loss < smallest_loss:
                        smallest_loss = loss
                        self.logger.log(logging.INFO, "New smallest loss after {samples:g} samples: {loss:g}! Save the weights...".format(loss=smallest_loss, samples=samples))
                        self.save_tf_weights(file_name, session)
                    else:
                        self.logger.log(logging.INFO, "Not smallest loss after {samples:g} samples: {loss:g}.".format(loss=loss, samples=samples))
                    self.config.stop_criterion.update(TrainingStatus(self.config, trained_batches, loss))
                    if self.config.stop_criterion.should_stop():
                        self.logger.log(logging.INFO, "Stop training loop")
                        break
                self.logger.log(logging.DEBUG, "Do one training step...")
                self.train_one_step(session)
                trained_batches += 1
