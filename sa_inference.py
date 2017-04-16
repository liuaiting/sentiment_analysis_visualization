# -*- coding: utf-8 -*-
"""
Created on Tue March 7 2017

@author: Aiting Liu

sa_interface :
    input: string, input sequence of words
    output: label value
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import json
import tensorflow as tf

import sa_model
import data_utils


tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 100, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("sent_vocab_size", 110000, "max vocab Size.")
tf.app.flags.DEFINE_integer("label_vocab_size", 4, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./log", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_sequence_length", 250,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.8,
                          "dropout keep cell input and output prob.")
FLAGS = tf.app.flags.FLAGS

sent_vocab_path = os.path.join(FLAGS.data_dir, "sent_vocab_%d.txt" % FLAGS.sent_vocab_size)
label_vocab_path = os.path.join(FLAGS.data_dir, "label.txt")

sent_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sent_vocab_path)
label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)

sent_vocab_size = len(sent_vocab)
label_vocab_size = len(label_vocab)


def load_sa_model():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    with tf.variable_scope("sentiment_analysis", reuse=None):
        model_test = sa_model.SaModel(
            session,
            sent_vocab_size, label_vocab_size, FLAGS.max_sequence_length,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            forward_only=True)
    model_test.saver.restore(session, tf.train.get_checkpoint_state(FLAGS.train_dir).model_checkpoint_path)

    return session, model_test


def sa_interface(sa_inputs, sess, model):
    """
    processing sa, get slot filling and label detection results.
    Args:
        sa_inputs: json string, contain a list of words.
        sess: session
        model: model from latest checkpint
    Return:
        sa_results: json string, slot filling result ang label detection results.
    """
    assert type(sa_inputs) == str
    # print("-"*30 + "sa MODULE" + "-"*30)
    inputs = json.loads(sa_inputs)["input"]
    # print("sa input: %s" % json.loads(sa_inputs))

    token_inputs = data_utils.sa_input_to_token_ids(inputs, sent_vocab_path, data_utils.naive_tokenizer)

    data = [[token_inputs, [0]]]
    # print(data)

    np_inputs, labels, sequence_length = model.get_one(data, 0)
    # print(np_inputs)

    _, loss, logits = model.step(np_inputs, labels, sequence_length, True)
    hyp_label = rev_label_vocab[np.argmax(logits[0])]

    sa_output = {"sa_result": {"label": hyp_label}}

    # print(sa_output)

    # # print("sa output: %s" % sa_output)
    # # print("write sa output in path: %s" % sa_result_file)
    # with tf.gfile.GFile(sa_result_file, 'w') as f:
    #     f.write(json.dumps(sa_output))
    #
    # # print("-"*30 + "END sa" + "-"*30)
    sa_output = json.dumps(sa_output)
    return sa_output

# Example.
sess, model = load_sa_model()
test = json.dumps({"input": ['还是', '记得', '要', '微笑']})

result = sa_interface(test, sess, model)
print(json.loads(result))


