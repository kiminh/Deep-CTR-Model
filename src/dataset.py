from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import utils

class Dataset(object):
    def __init__(self, filename, batch_size,
                 num_epochs, column_config,
                 bpe_model=None,  mode="train"):
        """get input function using dataset api"""
        self.filename = filename
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.column_config = column_config
        self.bpe_model = bpe_model
        self.mode = mode
        self.dataset = None
        self._unk_id = 1
        self.prefetch_buffer_size = 5000
        self.shuffle_buffer_size = 5000
       
    def _encode_as_ids(self, sent,  max_seq_len):
        if self.bpe_model :
            ids = self.bpe_model.encode_as_ids(sent, max_seq_len)
            ids = map(lambda x:  self._unk_id if x == 0 else x, ids) 
            return np.array(ids)
        else:
            return ValueError("no bpe model!!")

    def _encode_as_pieces(self, sent, max_seq_len):
        if self.bpe_model:
            sent = self.bpe_model.encode_as_pieces(sent, max_seq_len)
            return np.array(sent)
        elif self.char_split_enable:
            sent = list(sent)
        else:
            sent = sent.decode('utf-8').split()

        sent = sent[:max_seq_len]
        
        if len(sent) < max_seq_len:
            sent = sent + ['']*(max_seq_len-len(sent))

        #sent = [s.encode('utf-8') for s in sent]
        return np.array(sent)

    def input_fn(self):
        def parser_fn(record):
            csv_column_defaults = \
                self.column_config.column_defaults
            csv_delim = self.column_config.column_delimiter

            columns = tf.decode_csv(record,
                                    record_defaults=csv_column_defaults,
                                    field_delim=csv_delim,
                                    use_quote_delim=False,
                                    na_value="NULL")
            outputs = {}
            features = dict(zip(self.column_config.all_columns, columns))

            if self.mode == "train" or self.mode == "eval":
                y = tf.to_float(features.pop(
                    self.column_config.label_column))
            else:
                y = None
 
            for name in self.column_config.numeric_columns:
                outputs[name] = tf.to_float(features[name])
               
            for name in self.column_config.categorical_columns: # only category with int type
                outputs[name] = tf.to_int64(features[name])

            """
            if self.column_config.id_seq_column is not None:
                id_seq_column_name = self.column_config.id_seq_column
                max_seq_len = tf.constant(
                    self.column_config.get_property_for_id_seq_column(
                        "max_len"))

                outputs[id_seq_column_name] = tf.py_func(
                    self._encode_as_ids,
                    [features[id_seq_column_name],
                     max_seq_len],
                     [tf.int64])

                outputs[id_seq_column_name] = tf.reshape(
                    outputs[id_seq_column_name], [max_seq_len])
            """
            if y is not None:
                return outputs, y
            else:
                return outputs

        files = tf.data.Dataset.list_files(self.filename)
        self.dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TextLineDataset, cycle_length=5))
        if self.mode == "train":
            #self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)
            self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
                map_func=parser_fn, batch_size=self.batch_size))
            self.dataset = self.dataset.prefetch(buffer_size=self.prefetch_buffer_size)
            self.dataset = self.dataset.repeat(self.num_epochs)
            iterator = self.dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels
        else:
            self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
                map_func=parser_fn, batch_size=self.batch_size))
            self.dataset = self.dataset.prefetch(buffer_size=self.prefetch_buffer_size)
            self.dataset = self.dataset.repeat(self.num_epochs)
            iterator = self.dataset.make_one_shot_iterator()

            if self.mode == "eval":
                features, labels = iterator.get_next()
                return features, labels
            else:
                features = iterator.get_next()
                return features, None

