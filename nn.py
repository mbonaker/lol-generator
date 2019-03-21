import logging
from time import perf_counter
from typing import Optional

import tensorflow as tf
import numpy as np

import dataprovider
from config import ApplicationConfiguration

PURPOSE_TRAIN = 1
PURPOSE_TEST = 2
PURPOSE_PREDICT = 4


class NeuralNetwork:

    def __init__(self, data: dataprovider.DataProvider, config: ApplicationConfiguration):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data = data

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
        logit = tf.add(tf.matmul(w, x, True), b, "logit")
        predictions = tf.transpose(tf.concat((tf.nn.sigmoid(logit[:1, :]), logit[1:, :]), axis=0), name="predictions")
        self.layers.append((w, b, predictions))

        #
        # Create the evaluation of the DNN (loss/error and stuff)
        #
        error = tf.subtract(predictions, y, "error")
        self.mae = tf.reduce_mean(tf.abs(error), axis=0, name="MAE")
        self.train_summaries = tf.summary.merge([
            tf.summary.histogram("train_mean_error", self.mae),
        ])
        self.test_summaries = tf.summary.merge([
            tf.summary.histogram("test_mean_error", self.mae),
            tf.summary.scalar("test_error_duration", self.mae[0]),
        ])

        #
        # Create the optimization of the DNN (gradient and stuff)
        #
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        regularization = tf.constant(0, dtype=config.dtype)
        for w, _, _ in self.layers:
            if w is not None:
                regularization += tf.nn.l2_loss(w)
        regularization *= config.lambda_
        loss = tf.reduce_sum(self.mae) + regularization

        # grads_and_vars is a list of tuples (gradient, variable)
        parameters = [w for w, _, _ in self.layers if w is not None] + [b for _, b, _ in self.layers if b is not None]
        grads_and_vars = config.optimizer.compute_gradients(loss, var_list=parameters)
        capped_grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
        # apply the capped gradients.
        self.minimizer = config.optimizer.apply_gradients(capped_grads_and_vars, global_step=self.global_step)

        self.train_handle = None
        self.test_handle = None
        self.tf_writer: Optional[tf.summary.FileWriter] = None
        self.running_duration = 0

    def make_train_ndarray_batch_generator(self):
        def generate_ndarray_batches():
            ndarray = self.data.get_ndarray()[0:-self.config.test_data_amount, :]
            for batch in np.split(ndarray, np.arange(0, ndarray.shape[0], self.config.batch_size), axis=0):
                x = batch[:, self.data.np_structure.known_slice]
                y = batch[:, self.data.np_structure.unknown_without_win_slice]
                yield (x, y)
        return generate_ndarray_batches

    def make_test_ndarray_generator(self):
        def generate_ndarray():
            ndarray = self.data.get_ndarray()[-self.config.test_data_amount:, :]
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

    def train(self, sess: tf.Session):
        self.logger.log(logging.DEBUG, "Start training...")
        time_before = perf_counter()
        sess.run(self.minimizer, feed_dict={self.iterator_handle: self.train_handle})
        self.running_duration += perf_counter() - time_before
        self.logger.log(logging.DEBUG, "Training done.")

    def train_eval(self, sess: tf.Session):
        self.logger.log(logging.DEBUG, "Start training evaluation...")
        step, summaries = sess.run([self.global_step, self.train_summaries], feed_dict={self.iterator_handle: self.train_handle})
        self.logger.log(logging.DEBUG, "Training evaluation done.")
        self.tf_writer.add_summary(summaries, step)

    def test_eval(self, sess: tf.Session):
        self.logger.log(logging.DEBUG, "Start test evaluation...")
        step, summaries = sess.run([self.global_step, self.test_summaries], feed_dict={self.iterator_handle: self.test_handle})
        self.logger.log(logging.DEBUG, "Test evaluation done.")
        self.tf_writer.add_summary(summaries, step)

    def stop_session(self):
        self.tf_writer.close()
        tf.reset_default_graph()
