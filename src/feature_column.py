from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import utils
import rnn_column_dense
from json_config_parser import ColumnConfig, DeepFMColumnConfig


def _log_transformation(x):
    #x = utils.tf_print(x, "x=:")
    float_x = tf.to_float(x)
    return tf.log1p(float_x)


def _get_embedding_size(vocab_size):
    return int(math.floor(6*vocab_size**0.25))


class FeatureColumnInfo(object):
    def __init__(self, column_config_file, embedding_size_func=None):
        self.column_config = self._load_column_config(column_config_file)
        self.embedding_size_func = embedding_size_func

    def _load_column_config(self, column_config_file):
        return ColumnConfig(column_config_file)

    @property
    def feature_columns(self):
        feature_columns = []

        # numeric columns
        for col_name in self.column_config.continous_columns:
            real_column = tf.feature_column.numeric_column(
                col_name)
            feature_columns.append(real_column)
        
        # categorical columns(all categoircal columns should be int)
        for col_name in self.column_config.categorical_columns:
            hash_size = column_config.categorical_column_hash_size(col_name)
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
              col_name, hash_bucket_size=hash_size, dtype=tf.int64)

            if embedding_size_func is not None:
                embedding_dim = embedding_size_func(hash_size)
                categorical_column = tf.feature_column.embedding_column(
                    categorical_column, embedding_dim)
 
            feature_columns.append(categorical_column)
                   
        return feature_columns


class DeepCrossFeatureColumnInfo(FeatureColumnInfo):
    def __init__(self, column_config_file):
        super(DeepCrossFeatureColumnInfo, self).__init__(column_config_file,
            _get_embedding_size)

    def _load_column_config(self, column_config_file):
        return DeepCrossColumnConfig(column_config_file)

    @property
    def position_bias_feature_column(self):
        col_name = self.column_config.position_bias_column
        return [tf.feature_column.numeric_column(col_name)]


class DeepFMFeatureColumnInfo(FeatureColumnInfo):
    def __init__(self, column_config_file):
        super(DeepFMFeatureColumnInfo, self).__init__(column_config_file,
            _get_embedding_size)

        self.categorical_columns = self._get_categorical_feature_columns()

    def _load_column_config(self, column_config_file):
        return DeepFMColumnConfig(column_config_file)

    @property
    def position_bias_feature_column(self):
        col_name = self.column_config.position_bias_column
        return [tf.feature_column.numeric_column(col_name)]

    @property
    def bucketized_feature_column(self):
       columns = []
       for col_name, boundary in  self.column_config.bucketized_columns:
           bucketized_feature_column = tf.feature_column.bucketized_column(
               tf.feature_column.numeric_column(col_name),
               boundary)
           columns.append(bucketized_feature_column)
       return columns
  
    @property
    def numeric_feature_columns(self):
        columns = []
        for col_name in self.column_config.continous_columns:
            real_column = tf.feature_column.numeric_column(
                col_name)
            columns.append(real_column)
      
        for col_name in self.column_config.log_continous_columns:
            real_column = tf.feature_column.numeric_column(
                col_name,
                normalizer_fn=tf.log1p)
            columns.append(real_column)

        return columns

     
    def _get_categorical_feature_columns(self):
        columns = []
        for col_name in self.column_config.categorical_columns:
            hash_size = \
                self.column_config.categorical_column_hash_size(col_name)
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
               col_name, hash_bucket_size=hash_size, dtype=tf.int64)
            columns.append(categorical_column)
        return columns

    @property
    def categorical_feature_columns(self):
        return self.categorical_columns

    def categorical_embedding_feature_columns(self, embedding_size):
        columns = []
        for categorial_column in self.categorical_columns:
            columns.append(tf.feature_column.embedding_column(categorial_column, 
              embedding_size))
        return columns


