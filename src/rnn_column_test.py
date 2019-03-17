from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
import os
import tempfile

import numpy as np

from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver

