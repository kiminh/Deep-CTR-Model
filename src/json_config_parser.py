#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import sys
import six
import json
import codecs
import mmap
#import apply_bpe
from itertools import chain, combinations
from tensorflow.contrib.learn.python.learn import run_config


#
# onfiguration class for json file
#
class JsonConfig(object):
    def __init__(self, json_file):
        self.json_obj = _load_json_file(json_file)

    def _get(self, key, default=None):
        try :
            return self.json_obj[key]
        except KeyError:
            return default

def _load_json_file(json_file):
    try:
        print("json_file loading...", json_file)
        json_fp = open(json_file, 'r')
        config_dict = json.load(json_fp)
        return config_dict
    except ValueError as e:
        raise e
    except:
        raise RuntimeError("json parsing error")

#
# model configuration
#
class ModelConfig(JsonConfig):
    def __init__(self, model_config_file):
        super(ModelConfig, self).__init__(model_config_file)
    

class DeepFMModelConfig(ModelConfig):
    def __init__(self, model_config_file):
        super(DeepFMModelConfig, self).__init__(model_config_file)

        self.model_params = self._get("model_params", None)
        if self.model_params is None:
            raise ValueError("no model_params")

    @property
    def hidden_units(self):
        if "hidden_units" in self.model_params:
            return self.model_params["hidden_units"]
        else:
            raise ValueError("no hidden_units")

    @property
    def embedding_size(self): 
        if "embedding_size" in self.model_params:
            return int(self.model_params["embedding_size"])
        else:
            raise ValueError("no hidden_units")

    @property
    def dropout_rate(self):
        if "dropout_rate" in self.model_params:
            return float(self.model_params["dropout_rate"])
        else:
            return None

class DeepCrossModelConfig(ModelConfig):
    def __init__(self, model_config_file):
        super(DeepCrossModelConfig, self).__init__(model_config_file)
 
#
# feature column configuration
#
class ColumnConfig(JsonConfig):
    def __init__(self, json_file):
        super(ColumnConfig, self).__init__(json_file)

    @property
    def label_column(self, label_name="label"):
        return self._get(label_name, None)

    @property
    def all_columns(self):
        _column_defaults = self._get("column_defaults") # [(column_name, default_value)]
        if _column_defaults is None:
            raise ValueError("no column_defaults")
        return [ c[0] for c in _column_defaults]

    @property
    def column_defaults(self):
        _column_defaults = self._get("column_defaults") # [(column_name, default_value)]
        if _column_defaults is None:
            raise ValueError("no column_defaults")
        return [ c[1] for c in _column_defaults]

    @property
    def column_delimiter(self):
        return self._get("column_delimiter", ",")

    @property
    def continous_columns(self):
        return self._get("continous_columns", [])


    @property
    def log_continous_columns(self):
        return self._get("log_continous_columns", [])

    def categorical_column_hash_size(self, column_name):
        hash_sizes = self._get("categorical_column_hash_sizes")
        if column_name in hash_sizes:
            return hash_sizes[column_name]
        else:
            raise ValueError("no hash size for {}".format(column_name))

    @property
    def categorical_columns(self):
        return self._get("categorical_columns", [])

    @property 
    def numeric_columns(self):
        return self.continous_columns + self.log_continous_columns

    @property
    def bucketized_columns(self):
        columns = self._get("bucketized_columns", [])
        return [(c["col_name"], c["boundary"]) for column  in columns]
            

class DeepFMColumnConfig(ColumnConfig):
    def __init__(self, json_file):
        super(DeepFMColumnConfig, self).__init__(json_file)

    @property
    def position_bias_column(self):
        position_bias_column_name = self._get("position_bias_column", None)
        if position_bias_column_name  is None:
            raise ValueError("no positional bais column!")
        return position_bias_column_name

    @property
    def numeric_columns(self):
        return self.continous_columns + self.log_continous_columns + [self.position_bias_column]

    @property
    def log_continous_columns(self):
        return self._get("log_continous_columns", [])


class DeepCrossColumnConfig(DeepFMColumnConfig):
    def __init__(self, json_file):
        super(DeepCrossColumnConfig, self).__init__(json_file)

    

def load_model_config(model_type, model_config_file):
    if model_type == "deepfm":
        return DeepFMModelConfig(model_config_file)
    elif model_type == "deepcross":
        return DeepCrossModelConfig(model_config_file)
    else:
        raise ValueError("no such model config")
       

