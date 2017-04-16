# -*- coding: utf-8 -*-
"""
Created on Thur Mar 2 2017

@author: Aiting Liu

Multi-task RNN model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import data_utils


class SaModel(object):
    """Wait for completing ......"""
    def __init__(self,
                 session,
                 sent_vocab_size,
                 label_vocab_size,
                 max_sequence_length,
                 word_embedding_size,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 # learning_rate_decay_factor,
                 dropout_keep_prob=0.8,
                 use_lstm=True,
                 # num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
            session: session
            sent_vocab_size: int, size of the source sentence vocabulary.
            label_vocab_size: int, size of the label label vocabulary. dummy, only one label.
            max_sequence_length: int, specifies maximum input length.
                Training instances' inputs will be padded accordingly.
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            # learning_rate_decay_factor: decay learning rate by this much when needed.
            use_lstm: if true, we use LSTM cells instead of GRU cells.
            # num_samples: number of samples for sampled softmax.
            forward_only: if set, we do not construct the backward pass in the model.
            dtype: the data type to use to store internal variables.
        """
        self.session = session
        self.sent_vocab_size = sent_vocab_size
        self.label_vocab_size = label_vocab_size
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size

        with tf.name_scope("inputs"):
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype, name='learning_rate')
            # self.learning_rate_decay_op = self.learning_rate.assign(
            #     self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            # Feeds for inputs.
            with tf.name_scope("sequence_length_placeholder"):
                self.sequence_length = tf.placeholder(tf.int32, [None], name="mask")
            with tf.name_scope("input_placeholder"):
                self.inputs = tf.placeholder(tf.int32, shape=[None, self.max_sequence_length], name="input")
            with tf.name_scope("label_placeholder"):
                self.labels = tf.placeholder(tf.float32, shape=[None, self.label_vocab_size], name="label")

        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", [sent_vocab_size, word_embedding_size])
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding, self.inputs)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = self.embedding.name

            embedding.metadata_path = './log/embedding_meta.tsv'
            summary_writer = tf.summary.FileWriter('./log/')
            projector.visualize_embeddings(summary_writer, config)

        # Create the internal multi-layer cell for our RNN.
        with tf.name_scope("lstm_model"):
            single_cell = tf.contrib.rnn.GRUCell(size)
            if use_lstm:
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units=size, state_is_tuple=True)
            cell = single_cell
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

            if not forward_only and dropout_keep_prob < 1.0:
                with tf.name_scope("dropout"):
                    cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                         input_keep_prob=dropout_keep_prob,
                                                         output_keep_prob=dropout_keep_prob)
            # Training outputs and final state.
            self.outputs, self.state = tf.nn.dynamic_rnn(
                cell, self.embedded_inputs, sequence_length=self.sequence_length, dtype=dtype)

        # get the last time step output.
        # output = tf.transpose(self.outputs, [1, 0, 2])
        # self.last = tf.gather(output, int(output.get_shape()[0]) - 1)

            self.label_last = self.state.h  # shape batch_size, hidden_size

        with tf.name_scope("output_projection"):
            # Output projection.
            with tf.name_scope("weights"):
                label_weight = tf.get_variable('label_weight', shape=[size, label_vocab_size], dtype=dtype)
            with tf.name_scope("bias"):
                label_bias = tf.get_variable('label_bias', shape=[label_vocab_size], dtype=dtype)
            with tf.name_scope("xw_plus_b"):
                # Training outputs.
                self.label_outputs = tf.nn.xw_plus_b(self.label_last, label_weight, label_bias)

        with tf.name_scope("logits"):
            # Training logits.
            self.label_logits = tf.nn.softmax(self.label_outputs)

        with tf.name_scope("cross_entropy"):
            # Training cross entropy.
            self.label_crossent = -tf.reduce_sum(self.labels * tf.log(self.label_logits))
        # self.label_crossent = tf.nn.softmax_cross_entropy_with_logits(logits=self.label_outputs, labels=self.labels)

        with tf.name_scope("loss"):
            # Training loss.
            self.label_loss = tf.reduce_sum(self.label_crossent) / tf.cast(batch_size, tf.float32)

            self.losses = self.label_loss
            tf.summary.scalar('loss', self.losses)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            with tf.name_scope('train'):
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                gradients = tf.gradients(self.losses, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norm = norm
                self.update = opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step)

        with tf.name_scope("save"):
            self.saver = tf.train.Saver(tf.global_variables())
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./log', session.graph)
            self.test_writer = tf.summary.FileWriter('./log', session.graph)

    # @property
    # def input(self):
    #     return self.inputs
    #
    # @property
    # def output(self):
    #     return self.outputs
    #
    # @property
    # def slot_logit(self):
    #     return self.slot_logits
    #
    # @property
    # def label_logit(self):
    #     return self.label_logits

    def step(self, inputs, labels, sequence_length, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            # session: tensorflow session to use.
            inputs: list of numpy int vectors to feed as encoder inputs.
            labels: numpy float vectors to feed as target label label with shape=[batch_size, label_vocab_size].
            sequence_length: numpy float vectors to feed as sequence real length with shape=[batch_size, ].
            forward_only: whether to do the backward step or only forward.
            # merged: summary

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length/shape of inputs, labels, disagrees
            with the expected length/shape.
        """
        # Check if the sizes match.
        # input_size = self.max_sequence_length
        # if len(inputs) != input_size:
        #     raise ValueError("Inputs length must be equal to the config max sequence length,"
        #                      " %d != %d." % (len(inputs), input_size))
        # if labels.shape != (self.batch_size, self.label_vocab_size):
        #     raise ValueError("labels.shape must be equal to the expected shape.")

        # Input feed: inputs, labels, sequence_length as provided.
        input_feed = dict()
        input_feed[self.sequence_length.name] = sequence_length
        input_feed[self.inputs.name] = inputs
        input_feed[self.labels.name] = labels

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.merged,
                           self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.losses,  # Loss for this batch.
                           self.label_logits]  # Output logits.

        else:
            output_feed = [self.losses,  # Loss for this batch.
                           self.label_logits]  # Loss for this batch.

        outputs = self.session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[0], outputs[3], outputs[4]  # merged_summary, loss, outputs.
        else:
            return None, outputs[0], outputs[1]  # merged_summary, loss, outputs.
        # return input_feed, output_feed

    def get_batch(self, data):
        """Get a random batch of data from the data, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            self: get some configure
            data: a list in which each element contains
                lists of pairs of input and output data that we use to create a batch.

        Returns:
          The triple (inputs, s_attrs, s_locs, s_names, s_opes, s_ways, labels,
          sequence_length) for the constructed batch that has the proper format
          to call step(...) later.
        """
        input_size = self.max_sequence_length
        label_size = self.label_vocab_size

        inputs, labels = [], []
        batch_sequence_length_list = list()

        # Get a random batch of inputs, targets and labels from data,
        # pad them if needed.
        for _ in range(self.batch_size):
            _input, _label = random.choice(data)
            if len(_input) > input_size:
                batch_sequence_length_list.append(input_size)
                inputs.append(list(_input[:input_size]))
            else:
                batch_sequence_length_list.append(len(_input))

                # Inputs are padded.
                input_pad = [data_utils.PAD_ID] * (input_size - len(_input))
                inputs.append(list(_input + input_pad))

            # labels don't need padding.
            labels.append(_label)

        # Now we create batch-major vectors from the data selected above.
        batch_inputs = np.array(inputs, dtype=np.int32)

        def one_hot(vector, num_classes):
            assert isinstance(vector, np.ndarray)
            assert len(vector) > 0

            if num_classes is None:
                num_classes = np.max(vector) + 1
            else:
                assert num_classes > 0
                assert num_classes >= np.max(vector)

            result = np.zeros(shape=(len(vector), num_classes))
            result[np.arange(len(vector)), vector] = 1
            return result.astype(int)

        batch_labels = one_hot(np.array([labels[batch_idx][0] for batch_idx in range(self.batch_size)],
                                        dtype=np.int32), label_size)

        batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)

        return batch_inputs, batch_labels, batch_sequence_length

    def get_one(self, data, sample_id):
        """Get a single sample data from data, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            self: get some configure.
            data: a list in which each element contains lists of pairs of input
                and output data that we use to create a batch.
            sample_id: integer, which sample to get the batch for.

        Returns:
            The tuple (inputs, s_attrs, s_locs, s_names, s_opes, s_ways, labels,
            sequence_length) for the constructed batch that has the proper format
             to call step(...) later.
        """
        input_size = self.max_sequence_length
        label_size = self.label_vocab_size

        inputs, labels = [], []
        batch_sequence_length_list = list()

        # Get a random batch of inputs, targets and labels from data,
        # pad them if needed.
        _input, _label = data[sample_id]
        if len(_input) > input_size:
            batch_sequence_length_list.append(input_size)
            inputs.append(list(_input[:input_size]))
        else:
            batch_sequence_length_list.append(len(_input))

            # Inputs are padded.
            input_pad = [data_utils.PAD_ID] * (input_size - len(_input))
            inputs.append(list(_input + input_pad))

        # labels don't need padding.
        labels.append(_label)

        # Now we create batch-major vectors from the data selected above.
        batch_inputs = np.array(inputs, dtype=np.int32)

        def one_hot(vector, num_classes):
            assert isinstance(vector, np.ndarray)
            assert len(vector) > 0

            if num_classes is None:
                num_classes = np.max(vector) + 1
            else:
                assert num_classes > 0
                assert num_classes >= np.max(vector)

            result = np.zeros(shape=(len(vector), num_classes))
            result[np.arange(len(vector)), vector] = 1
            return result.astype(int)

        batch_labels = one_hot(np.array([labels[batch_idx][0] for batch_idx in range(1)],
                                        dtype=np.int32), label_size)

        batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)

        return batch_inputs, batch_labels, batch_sequence_length
