import itertools
import logging
import io
from collections import OrderedDict
from time import perf_counter
from typing import Optional, TextIO, List, Tuple

import tensorflow as tf
import numpy as np

import dataprovider as dp
from config import ApplicationConfiguration
from config import TrainingStatus
from nn import Generator
from nn import Discriminator

PURPOSE_TRAIN = 1
PURPOSE_TEST = 2
PURPOSE_PREDICT = 4


class TrainableGenerator(Generator):
    def __init__(self, field_structure: dp.FieldStructure, column_structure: dp.ColumnStructure, config: ApplicationConfiguration):
        super().__init__(field_structure, column_structure, config)

        self.unknown_field_structure = dp.FieldStructure(field_structure.data_path, dp.PORTION_UNKNOWN - dp.PORTION_WIN, True)
        self.unknown_column_structure = dp.ColumnStructure(self.unknown_field_structure, self.config.dtype, dp.PORTION_UNKNOWN - dp.PORTION_WIN, True)

        self.random_state = np.random.RandomState(config.seed)

        self.logger.log(logging.DEBUG, "Initialize layers for trainable generator")
        self.layers = self.make_layers(len(column_structure.known_indices), len(column_structure.unknown_without_win_indices))

        self.need_tf_weight_reassignment = False
        self.best_seen_evaluation_accuracy = None

    def reassign_tf_weights_if_necessary(self, sess: tf.Session):
        if self.need_tf_weight_reassignment:
            self.logger.log(logging.DEBUG, "Reassign the tensorflow weights.")
            for (np_w, np_b), (tf_w, tf_b, _) in zip(zip(self.weights, self.biases), (layer for layer in self.layers if layer[0] is not None)):
                sess.run(tf.assign(tf_w, np_w))
                sess.run(tf.assign(tf_b, np_b))

    def make_minimizer(self, loss: tf.Tensor) -> Tuple[tf.Operation, tf.Variable]:
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # grads_and_vars is a list of tuples (gradient, variable)
        parameters = [w for w, _ in self.layers if w is not None] + [b for _, b in self.layers if b is not None]
        grads_and_vars = self.config.optimizer(self.config.learning_rate).compute_gradients(loss, var_list=parameters)
        capped_grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
        # apply the capped gradients.
        minimizer = self.config.optimizer(self.config.learning_rate).apply_gradients(capped_grads_and_vars, global_step=global_step)
        return minimizer, global_step

    def make_summaries(self, predictions: tf.Tensor, ground_truth: tf.Tensor, accuracy: tf.Tensor):
        error = tf.subtract(predictions, ground_truth, "error")
        train_summaries = tf.summary.merge([
            # tf.summary.histogram("train_mean_error", self.mae),
            tf.summary.scalar("train_accuracy", tf.reduce_mean(accuracy)),
        ])
        test_evaluation_summaries = [
            tf.summary.scalar("test_accuracy", tf.reduce_mean(accuracy)),
        ]
        for field in self.field_structure.unknown_without_win:
            column_slice = self.unknown_column_structure.field_to_column_slice(field)
            column_mae = tf.reduce_mean(tf.abs(error[:, column_slice]))
            error_summary = tf.summary.scalar("test_error_{column_name:s}".format(column_name=field.name), column_mae)
            test_evaluation_summaries.append(error_summary)
            column_prediction = predictions[:, column_slice]
            prediction_summary = tf.summary.histogram("test_prediction_{column_name:s}".format(column_name=field.name), tf.reduce_mean(column_prediction, axis=1))
            test_evaluation_summaries.append(prediction_summary)
        test_summaries = tf.summary.merge(test_evaluation_summaries)
        return train_summaries, test_summaries

    def make_data_iterator(self, data: dp.DataProvider) -> Tuple[tf.data.Iterator, tf.data.Iterator, tf.placeholder, tf.data.Iterator]:
        train_data_shape = (
            (None, data.known.shape[1]),
            (None, data.unknown_without_win.shape[1]),
        )
        train_data = tf.data.Dataset.from_generator(
            self.make_train_ndarray_batch_generator(data),
            output_types=(self.config.dtype, self.config.dtype),
            output_shapes=train_data_shape,
        )
        train_data = train_data.repeat()
        train_data = train_data.prefetch(1)
        train_data = train_data.apply(tf.data.experimental.unbatch())
        train_data = train_data.shuffle(self.config.batch_size)
        train_data = train_data.batch(self.config.batch_size, drop_remainder=False)
        test_data_shape = (
            (None, data.known.shape[1]),
            (None, data.unknown_without_win.shape[1]),
        )
        test_data = tf.data.Dataset.from_generator(
            self.make_test_ndarray_generator(data),
            output_types=(self.config.dtype, self.config.dtype),
            output_shapes=test_data_shape,
        )
        test_data = test_data.repeat()
        train_iterator = train_data.make_initializable_iterator()
        test_iterator = test_data.make_initializable_iterator()
        iterator_handle = tf.placeholder(tf.string, shape=[])
        data_iterator = tf.data.Iterator.from_string_handle(iterator_handle, train_data.output_types, train_data.output_shapes)
        return train_iterator, test_iterator, iterator_handle, data_iterator

    def make_metrics(self, logit: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        def ignore_col(col: dp.FieldSpecification) -> bool:
            return col.name in self.config.ignored_columns

        losses = OrderedDict()
        labels = OrderedDict()
        accuracies = OrderedDict()
        accuracy_amount = 0
        for data_slice in self.unknown_column_structure.generate_slices(lambda c: not ignore_col(c) and c.handling == dp.FieldSpecification.HANDLING_CONTINUOUS):
            ys = y[:, data_slice]
            logit_slice = logit[:, data_slice]
            loss = tf.reduce_mean(tf.reduce_sum(tf.abs(ys - logit_slice), axis=1))
            # loss = tf.cast(tf.losses.huber_loss(ys, logit_slice, delta=1.0, reduction=tf.losses.Reduction.MEAN), self.config.dtype) * 10
            losses['continuous_{!r}'.format(data_slice)] = loss
            accuracies['continuous_{!r}'.format(data_slice)] = tf.reduce_sum((tf.abs(ys - logit_slice) + 1) ** -1, axis=1)
            accuracy_amount += tf.shape(logit_slice)[1]
        for field in self.field_structure.unknown_without_win:
            data_slice = self.unknown_column_structure.field_to_column_slice(field)
            ys = y[:, data_slice]
            logit_slice = logit[:, data_slice]
            handling = field.handling if field.name not in self.config.ignored_columns else dp.FieldSpecification.HANDLING_NONE
            if handling == dp.FieldSpecification.HANDLING_BOOL:
                loss = tf.cast(tf.losses.hinge_loss(labels=ys, logits=logit_slice, reduction=tf.losses.Reduction.MEAN), self.config.dtype)
                # loss = tf.cast(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=logit_slice)), self.config.dtype)
                losses['bool_{:s}'.format(field.name)] = loss
                labels['bool_{:s}'.format(field.name)] = ys
                accuracies['bool_{:s}'.format(field.name)] = tf.reduce_sum(tf.cast(tf.equal(logit_slice > 0, ys > 0), self.config.dtype), axis=1)
                accuracy_amount += tf.shape(logit_slice)[1]
        # per participant one-hot encodings
        for pid in range(10):
            for key_part in ("highestAchievedSeasonTier", "timeline.lane", "timeline.role"):
                key = 'participants.{id:d}.{k:s}'.format(id=pid, k=key_part)
                slice_ = self.unknown_column_structure.field_name_to_column_slice(key)
                logit_slice = logit[:, slice_]
                ys = y[:, slice_]
                loss = tf.cast(tf.losses.hinge_loss(labels=ys, logits=logit_slice, reduction=tf.losses.Reduction.MEAN), self.config.dtype)
                losses['onehot_{:s}'.format(key)] = loss
                labels['onehot_{:s}'.format(key)] = ys
                accuracies['onehot_{:s}'.format(key)] = tf.cast(tf.equal(tf.argmax(logit_slice, 1, output_type=tf.int32), tf.argmax(ys, 1, output_type=tf.int32)), self.config.dtype)
                accuracy_amount += 1
        for team_id in (0, 1):
            ban_slices = []
            for ban_id in range(5):
                ban_name = "teams.{tid:d}.bans.{bid:d}.championId".format(tid=team_id, bid=ban_id)
                ban_slice = self.unknown_column_structure.field_name_to_column_slice(ban_name)
                ban_slices.append(ban_slice)
                logit_slice = logit[:, ban_slice]
                ys = y[:, ban_slice]
                accuracies["onehot_teams.{tid:d}.bans.X.championId".format(tid=team_id)] = tf.cast(tf.equal(tf.argmax(logit_slice, 1, output_type=tf.int32), tf.argmax(ys, 1, output_type=tf.int32)), self.config.dtype)
                accuracy_amount += 1
            logit_slice = sum(logit[:, ban_slice] for ban_slice in ban_slices)
            ys = sum(y[:, ban_slice] for ban_slice in ban_slices)
            loss = tf.cast(tf.losses.hinge_loss(labels=ys, logits=logit_slice, reduction=tf.losses.Reduction.MEAN), self.config.dtype)
            # loss = tf.cast(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=logit_slice)), self.config.dtype)
            losses["onehot_teams.{tid:d}.bans.X.championId".format(tid=team_id)] = loss
            labels["onehot_teams.{tid:d}.bans.X.championId".format(tid=team_id)] = ys
        if self.config.lambda_:
            regularization = sum(tf.nn.l2_loss(w) for w, _ in self.layers if w is not None) * self.config.lambda_
            losses["regularization"] = regularization
        loss = tf.math.add_n(list(losses.values()), "loss")
        accuracy = tf.math.divide(tf.math.add_n(list(accuracies.values())), tf.cast(accuracy_amount, tf.float16), name="accuracy")
        return loss, accuracy

    def make_layers(self, n_input: int, n_output: int) -> List[Tuple[tf.Variable, tf.Variable]]:
        layers = []
        previous_layer_size = n_input
        for i, l in enumerate(itertools.chain(self.config.hidden_layer_structure, (n_output,))):
            self.logger.log(logging.DEBUG, "Initialize weights with size {x:d}×{y:d}={s:d}".format(x=previous_layer_size, y=l, s=previous_layer_size*l))
            w = tf.Variable(tf.random_normal((previous_layer_size, l), 0, 0.05, dtype=self.config.dtype, seed=self.config.seed), name="weights_" + str(i))
            b = tf.Variable(tf.random_normal((l, 1), 0, 0.05, dtype=self.config.dtype, seed=self.config.seed), name="bias_" + str(i))
            layers.append((w, b))
            previous_layer_size = l
        return layers

    def make_logit(self, features: tf.Tensor) -> tf.Tensor:
        x = tf.cast(tf.transpose(features), self.config.dtype)
        apply_activation_before_forward = False
        x_is_sparse = True
        for w, b in self.layers:
            if apply_activation_before_forward:
                x = self.config.activation_function(x)
            x = tf.add(tf.matmul(w, x, True, b_is_sparse=x_is_sparse), b)
            apply_activation_before_forward = True
            x_is_sparse = False
        return tf.transpose(x)

    def make_output(self, logit: tf.Tensor) -> tf.Tensor:
        prediction_slices = []
        for data_slice, handling in self.unknown_column_structure.generate_handling_slices(self.config.ignored_columns):
            logit_slice = logit[:, data_slice]
            if handling == dp.FieldSpecification.HANDLING_NONE:
                y_hat_slice = tf.zeros_like(logit_slice)
            elif handling == dp.FieldSpecification.HANDLING_ONEHOT or handling == dp.FieldSpecification.HANDLING_BOOL:
                y_hat_slice = tf.nn.sigmoid(logit_slice)
            elif handling == dp.FieldSpecification.HANDLING_CONTINUOUS:
                y_hat_slice = logit_slice
            else:
                raise NotImplementedError()
            prediction_slices.append(y_hat_slice)
        return tf.concat(prediction_slices, axis=1)

    def make_train_ndarray_batch_generator(self, data: dp.DataProvider):
        def generate_ndarray_batches():
            ndarray = data.get_ndarray()[0:-self.config.test_data_amount - self.config.validation_data_amount, :]
            duration = data.columns.field_name_to_column_slice("gameDuration")
            for batch in np.split(ndarray, np.arange(self.config.batch_size, ndarray.shape[0], self.config.batch_size), axis=0):
                batch = data.columns.shuffle_participants(batch, self.random_state)
                # filter out 'remakes' (games that have been dissolved early due to leaving participants)
                keep = batch[:, duration] > (15 * 60 - 1762.503) / 493.163
                keep = keep[:, 0]
                x: np.ndarray = batch[keep, data.columns.known_slice]
                y: np.ndarray = batch[keep, data.columns.unknown_without_win_slice]
                # take some optional information from x out (simulate incomplete user input)
                if isinstance(x, np.memmap) or not x.flags.writeable:
                    x_new = np.ndarray(x.shape, x.dtype)
                    x_new[:, :] = x
                    x = x_new
                x = data.columns.randomly_unspecify_optional_columns(x, self.random_state, 0.0, 0.0)
                yield (x, y)
        return generate_ndarray_batches

    def make_test_ndarray_generator(self, data: dp.DataProvider):
        def generate_ndarray():
            ndarray = data.get_ndarray()[-self.config.test_data_amount - self.config.validation_data_amount:-self.config.test_data_amount, :]
            # filter out 'remakes' (games that have been dissolved early due to leaving participants)
            duration = data.columns.field_name_to_column_slice("gameDuration")
            keep = ndarray[:, duration] > (15 * 60 - 1762.503) / 493.163
            keep = keep[:, 0]
            x = ndarray[keep, data.columns.known_slice]
            y = ndarray[keep, data.columns.unknown_without_win_slice]
            # take some optional information from x out (simulate incomplete user input)
            if isinstance(x, np.memmap) or not x.flags.writeable:
                x_new = np.ndarray(x.shape, x.dtype)
                x_new[:, :] = x
                x = x_new
            x = data.columns.randomly_unspecify_optional_columns(x, self.random_state, 0.1, 0)
            yield (x, y)
        return generate_ndarray

    def update_tf_weights_to_np_weights(self, sess: tf.Session):
        self.weights = sess.run([w for w, b in self.layers if w is not None])
        self.biases = sess.run([b for w, b in self.layers if b is not None])
        assert len(self.weights) == len(self.biases)

    def load_weights(self, file_name: str):
        super().load_weights(file_name)
        self.need_tf_weight_reassignment = True

    def save_tf_weights(self, file_name: str, sess: tf.Session):
        self.update_tf_weights_to_np_weights(sess)
        self.save_weights(file_name)

    def run_data_initializers(self, session: tf.Session, train_iterator: tf.data.Iterator, test_iterator: tf.data.Iterator):
        train_handle = session.run(train_iterator.string_handle())
        test_handle = session.run(test_iterator.string_handle())
        session.run(train_iterator.initializer)
        session.run(test_iterator.initializer)
        session.run(tf.global_variables_initializer())
        return train_handle, test_handle

    def get_best_seen_evaluation_accuracy(self) -> float:
        return self.best_seen_evaluation_accuracy

    def train(self, file_name: str, data: dp.DataProvider):
        train_iterator, test_iterator, iterator_handle, data_iterator = self.make_data_iterator(data)
        self.layers = self.make_layers(data.known.shape[1], data.unknown_without_win.shape[1])
        features, targets = data_iterator.get_next(name="layer_in")
        logit = self.make_logit(features)
        loss, accuracies = self.make_metrics(logit, targets)
        accuracy = tf.reduce_mean(accuracies)
        predictions = self.make_output(logit)
        train_summary, test_summary = self.make_summaries(predictions, targets, accuracy)
        minimizer, step = self.make_minimizer(loss)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            self.logger.log(logging.DEBUG, "Initialize the iterators")
            train_handle, test_handle = self.run_data_initializers(session, train_iterator, test_iterator)

            self.logger.log(logging.DEBUG, "Start the file writer for tensorboard")
            tf_writer = tf.summary.FileWriter(self.config.tensorboard_path, session.graph, flush_secs=10)

            trained_batches = 0
            samples_since_test_evaluation = 0
            samples_since_train_evaluation = 0
            self.logger.log(logging.DEBUG, "Start training loop")
            while True:
                samples = trained_batches * self.config.batch_size
                samples_since_test_evaluation += self.config.batch_size
                samples_since_train_evaluation += self.config.batch_size
                do_test_eval = samples == 0 or samples_since_train_evaluation >= self.config.samples_per_train_evaluation
                do_train_eval = do_test_eval or samples_since_train_evaluation >= self.config.samples_per_train_evaluation

                run_parameters = {
                    'minimizer': minimizer,
                    'step': step,
                }
                if do_train_eval:
                    self.logger.log(logging.DEBUG, "Evaluate the network on training data...")
                    run_parameters['accuracy'] = accuracy
                    run_parameters['summary'] = train_summary

                self.logger.log(logging.DEBUG, "Do one training step...")
                train_result = session.run(run_parameters, feed_dict={iterator_handle: train_handle})

                if do_train_eval:
                    samples_since_train_evaluation -= self.config.samples_per_train_evaluation
                    tf_writer.add_summary(train_result['summary'], train_result['step'])
                if do_test_eval:
                    samples_since_test_evaluation -= self.config.samples_per_test_evaluation
                    test_result = session.run({
                        'summary': test_summary,
                        'loss': loss,
                        'accuracy': accuracy,
                    }, feed_dict={iterator_handle: test_handle})
                    tf_writer.add_summary(test_result['summary'], train_result['step'])
                    if self.best_seen_evaluation_accuracy is None or self.best_seen_evaluation_accuracy < test_result['accuracy']:
                        self.best_seen_evaluation_accuracy = test_result['accuracy']
                        self.save_tf_weights(file_name, session)
                    self.logger.log(logging.INFO, "Samples: {samples:.4e} | Loss: {loss:.4e} | Accuracy: {accuracy:.4e} | Best accuracy: {bestaccuracy:.4e} | Train accuracy: {train_acc:.4e}".format(
                        loss=test_result['loss'],
                        samples=samples,
                        accuracy=test_result['accuracy'],
                        bestaccuracy=self.best_seen_evaluation_accuracy,
                        train_acc=train_result['accuracy'],
                    ))
                    self.config.stop_criterion.update(TrainingStatus(self.config, trained_batches, test_result['loss']))
                    if self.config.stop_criterion.should_stop():
                        self.logger.log(logging.INFO, "Stop training loop")
                        break
                trained_batches = train_result['step']
        tf_writer.close()
        tf.reset_default_graph()
