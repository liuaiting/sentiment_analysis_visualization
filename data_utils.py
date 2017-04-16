# -*-coding=utf-8-*-
"""
Created on Fri 7 Apr 2017

@author: Aiting Liu

Prepare data for RNN model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = []

PAD_ID = 0

UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1  # sequence labeling need padding (mask)
# UNK_ID_dict['no_padding'] = 0  # sequence classification

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(，。！？、：；（）])")
_DIGIT_RE = re.compile(r"\d")


# def basic_tokenizer(sentence):
#     """Very basic tokenizer: split the sentence into a list of tokens."""
#     words = []
#     for space_sepatated_fragment in sentence.strip().split():
#         words.extend(re.split(_WORD_SPLIT, space_sepatated_fragment))
#     return [w for w in words if w]


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    return sentence.split(' ')


def naive_tokenizer(sentence):
    """Naive tokenizer: split the sentence by space into a list of tokens."""
    return sentence.split()


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
    """Create vocabulary file (if if does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data file that will be used to created vocabulary.
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print(" processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = START_VOCAB_dict["with_padding"] + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with tf.gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
        vocabulary_path: path to the file containing the vocabulary.

    Returns:
        a pair: the vocabulary( a dictionary mapping string to integers), and
        the reversed vocabulary( a list, which reverse the vocabulary mapping).

    Raises:
            ValueError: if the provided vocabulary_path does not exist.
    """
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        # print(vocab)
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found." % vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, UNK_ID,
                          tokenizer=None, normalize_digits=False):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False, use_padding=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data-path, calls the above
    sentence_to_token_ids, and saves the result to target_path.See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not tf.gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with tf.gfile.GFile(data_path, mode="r") as data_file:
            with tf.gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print(" tokenizing line %d" % counter)
                    if use_padding:
                        UNK_ID = UNK_ID_dict["with_padding"]
                        token_ids = sentence_to_token_ids(line, vocab, UNK_ID, tokenizer,
                                                          normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                    else:
                        words = line.strip().split()
                        # print(words)
                        token_ids = [vocab.get(w) for w in words]
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def sa_input_to_token_ids(sa_inputs, vocabulary_path, tokenizer=None, normalize_digits=False, use_padding=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data-path, calls the above
    sentence_to_token_ids, and saves the result to target_path.See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        sa_inputs: a list of words
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
        use_padding:
    """
    inputs = ' '.join(sa_inputs)
    print("Tokenizing data in %s" % inputs)
    vocab, _ = initialize_vocabulary(vocabulary_path)

    if use_padding:
        UNK_ID = UNK_ID_dict["with_padding"]
    else:
        UNK_ID = UNK_ID_dict["no_padding"]
    token_ids = sentence_to_token_ids(inputs, vocab, UNK_ID, tokenizer,
                                      normalize_digits)

    return token_ids


def create_label_vocab(vocabulary_path, data_path):
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print(" processing line %d" % counter)
                label = line.strip()
                vocab[label] = 1
            label_list = sorted(vocab)
            with tf.gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for k in label_list:
                    # print(k)
                    vocab_file.write(k + "\n")


def prepare_sa_data(data_dir, sent_vocab_size):
    train_path = data_dir + "/train/train"
    valid_path = data_dir + "/valid/valid"
    test_path = data_dir + "/test/test"

    # Create vocabularies of the appropriate sizes.
    sent_vocab_path = os.path.join(data_dir, "sent_vocab_%d.txt" % sent_vocab_size)
    label_vocab_path = os.path.join(data_dir, "label.txt")

    create_vocabulary(sent_vocab_path, train_path + "_sent.txt", sent_vocab_size, tokenizer=naive_tokenizer)
    create_label_vocab(label_vocab_path, train_path + "_label.txt")

    # Create token ids for the training data.
    sent_train_ids_path = train_path + ("_ids%d_sent.txt" % sent_vocab_size)
    label_train_ids_path = train_path + "_ids_label.txt"  

    data_to_token_ids(train_path + "_sent.txt", sent_train_ids_path, sent_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + "_label.txt", label_train_ids_path, label_vocab_path, normalize_digits=False,
                      use_padding=False)

    # Create token ids for the development data.
    sent_valid_ids_path = valid_path + ("_ids%d_sent.txt" % sent_vocab_size)
    label_valid_ids_path = valid_path + "_ids_label.txt"

    data_to_token_ids(valid_path + "_sent.txt", sent_valid_ids_path, sent_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(valid_path + "_label.txt", label_valid_ids_path, label_vocab_path, normalize_digits=False,
                      use_padding=False)

    # Create token ids for the test data.
    sent_test_ids_path = test_path + ("_ids%d_sent.txt" % sent_vocab_size)
    label_test_ids_path = test_path + "_ids_label.txt"

    data_to_token_ids(test_path + "_sent.txt", sent_test_ids_path, sent_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + "_label.txt", label_test_ids_path, label_vocab_path, normalize_digits=False,
                      use_padding=False)

    return (sent_train_ids_path, label_train_ids_path,
            sent_valid_ids_path, label_valid_ids_path,
            sent_test_ids_path, label_test_ids_path,
            sent_vocab_path, label_vocab_path)


# def main():
#     prepare_multi_task_data('./data', 500)
#     print('Done prepare multi task data.')
#
#
# if __name__ == '__main__':
#     main()





















