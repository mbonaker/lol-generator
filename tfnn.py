import logging
import io
from time import perf_counter
from typing import Optional, TextIO

import tensorflow as tf
import numpy as np

import dataprovider as dp
from config import ApplicationConfiguration
from nn import NeuralNetwork

PURPOSE_TRAIN = 1
PURPOSE_TEST = 2
PURPOSE_PREDICT = 4


class TrainableNeuralNetwork(NeuralNetwork):

    def __init__(self, data: dp.DataProvider, config: ApplicationConfiguration):
        super().__init__(data, config)

        unknown_csv_structure = dp.CsvCorpusStructure(self.data.data_path, dp.PORTION_UNKNOWN - dp.PORTION_WIN)
        unknown_data_structure = dp.NumpyCorpusStructure(unknown_csv_structure, self.config.dtype, dp.PORTION_UNKNOWN - dp.PORTION_WIN)

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
            (config.test_data_amount, data.known.shape[1]),
            (config.test_data_amount, data.unknown_without_win.shape[1]),
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
            w = tf.Variable(
                tf.random_normal((previous_layer_size, l), 0, 0.05, dtype=config.dtype, seed=config.seed),
                name="weights_" + str(i))
            b = tf.Variable(tf.random_normal((l, 1), 0, 0.05, dtype=config.dtype, seed=config.seed),
                            name="bias_" + str(i))
            sum_w = tf.matmul(w, x, True, b_is_sparse=x_is_sparse)
            x_is_sparse = False
            x = tf.nn.relu(tf.add(sum_w, b), "layer_" + str(i))
            self.layers.append((w, b, x))
            previous_layer_size = l
        w = tf.Variable(
            tf.random_normal((previous_layer_size, target_amount), 0, 0.05, dtype=config.dtype, seed=config.seed),
            name="weights_end")
        b = tf.Variable(tf.random_normal((target_amount, 1), 0, 0.05, dtype=config.dtype, seed=config.seed),
                        name="bias_end")
        logit = tf.transpose(tf.add(tf.matmul(w, x, True), b), name="logit")

        #
        # Calculate metrics like error, predictions etc.
        #
        regularization = tf.constant(0, dtype=config.dtype)
        for w, _, _ in self.layers:
            if w is not None:
                regularization += tf.nn.l2_loss(w)
        regularization *= config.lambda_
        prediction_slices = []
        self.loss = 0
        for data_slice, handling in unknown_data_structure.generate_handling_slices():
            y_real = y[:, data_slice]
            y_pred = logit[:, data_slice]
            if handling == dp.CsvColumnSpecification.HANDLING_ONEHOT:
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=y_real, logits=y_pred, reduction=tf.losses.Reduction.MEAN)
                y_pred = tf.nn.sigmoid(y_pred)
            elif handling == dp.CsvColumnSpecification.HANDLING_BOOL:
                y_pred = tf.nn.sigmoid(y_pred)
                self.loss = tf.losses.log_loss(labels=y_real, predictions=y_pred, reduction=tf.losses.Reduction.MEAN)
            else:
                self.loss = tf.reduce_mean(tf.abs(y_real - y_pred))
            prediction_slices.append(y_pred)
        predictions = tf.concat(prediction_slices, axis=1)
        self.layers.append((w, b, predictions))

        #
        # Create the evaluation of the DNN (loss/error and stuff)
        #
        error = tf.subtract(predictions, y, "error")
        self.mae = tf.reduce_mean(tf.math.abs(error), axis=0, name="MAE")
        self.train_summaries = tf.summary.merge([
            # tf.summary.histogram("train_mean_error", self.mae),
            tf.summary.scalar("train_loss", self.loss),
        ])
        test_evaluation_summaries = [
            tf.summary.histogram("test_mean_error", self.mae),
            tf.summary.scalar("test_loss", self.loss),
        ]
        for csv_column in unknown_csv_structure.columns:
            column_slice = unknown_data_structure.csv_column_spec_to_np_slice(csv_column)
            column_mae = tf.reduce_mean(tf.abs(error[:, column_slice]))
            summary = tf.summary.scalar("test_error_{column_name:s}".format(column_name=csv_column.name), column_mae)
            test_evaluation_summaries.append(summary)
        self.test_summaries = tf.summary.merge(test_evaluation_summaries)


        #
        # Create the optimization of the DNN (gradient and stuff)
        #
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # grads_and_vars is a list of tuples (gradient, variable)
        parameters = [w for w, _, _ in self.layers if w is not None] + [b for _, b, _ in self.layers if b is not None]
        grads_and_vars = config.optimizer.compute_gradients(self.loss, var_list=parameters)
        capped_grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
        # apply the capped gradients.
        self.minimizer = config.optimizer.apply_gradients(capped_grads_and_vars, global_step=self.global_step)

        self.train_handle = None
        self.test_handle = None
        self.tf_writer: Optional[tf.summary.FileWriter] = None
        self.running_duration = 0
        self.need_tf_weight_reassignment = False

    def make_train_ndarray_batch_generator(self):
        def generate_ndarray_batches():
            ndarray = self.data.get_ndarray()[0:-self.config.test_data_amount - self.config.validation_data_amount, :]
            for batch in np.split(ndarray, np.arange(0, ndarray.shape[0], self.config.batch_size), axis=0):
                x = batch[:, self.data.np_structure.known_slice]
                y = batch[:, self.data.np_structure.unknown_without_win_slice]
                yield (x, y)
        return generate_ndarray_batches

    def make_test_ndarray_generator(self):
        def generate_ndarray():
            ndarray = self.data.get_ndarray()[-self.config.test_data_amount - self.config.validation_data_amount:-self.config.test_data_amount, :]
            x = ndarray[:, self.data.np_structure.known_slice]
            y = ndarray[:, self.data.np_structure.unknown_without_win_slice]
            yield (x, y)
        return generate_ndarray

    def start_session(self, sess: tf.Session):
        self.logger.log(logging.DEBUG, "Initialize the iterator")
        self.train_handle = sess.run(self.train_iterator.string_handle())
        self.test_handle = sess.run(self.test_iterator.string_handle())
        sess.run(self.train_iterator.initializer)
        sess.run(self.test_iterator.initializer)
        sess.run(tf.global_variables_initializer())

        self.tf_writer = tf.summary.FileWriter(self.config.tensorboard_path, sess.graph, flush_secs=10)

    def train_one_step(self, sess: tf.Session):
        self.reassign_tf_weights_if_necessary(sess)
        self.logger.log(logging.DEBUG, "Start training...")
        time_before = perf_counter()
        sess.run(self.minimizer, feed_dict={self.iterator_handle: self.train_handle})
        self.running_duration += perf_counter() - time_before
        self.logger.log(logging.DEBUG, "Training done.")

    def train_eval(self, sess: tf.Session):
        self.reassign_tf_weights_if_necessary(sess)
        self.logger.log(logging.DEBUG, "Start training evaluation...")
        step, summaries = sess.run([self.global_step, self.train_summaries], feed_dict={self.iterator_handle: self.train_handle})
        self.logger.log(logging.DEBUG, "Training evaluation done.")
        training_amount = step * self.config.batch_size
        self.tf_writer.add_summary(summaries, training_amount)

    def test_eval(self, sess: tf.Session) -> float:
        self.reassign_tf_weights_if_necessary(sess)
        self.logger.log(logging.DEBUG, "Start test evaluation...")
        step, summaries, loss = sess.run([
            self.global_step,
            self.test_summaries,
            tf.reduce_mean(self.loss),
        ], feed_dict={self.iterator_handle: self.test_handle})
        self.logger.log(logging.DEBUG, "Test evaluation done.")
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
            for i in range(0, self.config.training_amount // self.config.batch_size):
                training_amount = i * self.config.batch_size
                if training_amount % (1 << 16) == 0:
                    self.train_eval(session)
                if training_amount % (1 << 19) == 0:
                    loss = self.test_eval(session)
                    if loss < smallest_loss:
                        smallest_loss = loss
                        self.save_tf_weights(file_name, session)
                        self.logger.log(logging.INFO, "New smallest loss: {loss:g}".format(loss=smallest_loss))
                self.train_one_step(session)
