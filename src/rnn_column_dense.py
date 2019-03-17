import math
import collections
import utils

from tensorflow.contrib import layers

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.python.training import checkpoint_utils


class _RNNColumn(
    fc_core._DenseColumn,
    collections.namedtuple("_RNNColumn", [
        "categorical_column", "embedding_dimension", "max_sequence_length",
        "initializer", "num_units", "cell_type", "activation_fn",
        "bidirectional_rnn", "mode", "dropout_keep_probabilities",
        "ckpt_to_load_from", "tensor_name_in_ckpt", "shared_embedding_name",
        "shared_vocab_size", "max_norm", "trainable"])):

    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = '{}_rnn'.format(self.categorical_column.name)
        return self._name
       
    @property
    def config(self):
        return _get_feature_config(self.sparse_id_column)

    @property
    def _parse_example_spec(self):
        return self.categorical_column._parse_example_spec #pylint: disable=protected-access
    
    def _transform_feature(self, inputs):
        return inputs.get(self.categorical_column)

    @property
    def _variable_shape(self):
        if not hasattr(self, '_shape'):
            self._shape = tensor_shape.vector(self.num_units)
        return self._shape

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        #Get sparse IDs and weights.
        sparse_tensors = self.categorical_column._get_sparse_tensors( #pylint: disable=protected-access
            inputs, weight_collections=weight_collections, trainable=trainable)
        sparse_ids = sparse_tensors.id_tensor
        batch_size = sparse_ids.dense_shape[0]
        dense_tensor_ids = sparse_ops.sparse_to_dense(sparse_ids.indices,
                                                      [batch_size, 
                                                       self.max_sequence_length],
                                                      sparse_ids.values,
                                                      default_value=0)
        
        # Create embedding weight, and restore from checkpoint if necessary.
        embedding_weights = variable_scope.get_variable(
            name='embedding_weights',
            shape=(self.categorical_column._num_buckets, self.embedding_dimension),  # pylint: disable=protected-access
            dtype=dtypes.float32,
            initializer=self.initializer,
            trainable=self.trainable and trainable,
            collections=weight_collections)
        if self.ckpt_to_load_from is not None:
            to_restore = embedding_weights
            if isinstance(to_restore, variables.PartitionedVariable):
                to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
            checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {
                self.tensor_name_in_ckpt: to_restore
            })

        #dense_tensor_ids = utils.tf_print(dense_tensor_ids, "dense:")
        embedding_inputs = embedding_lookup(
            embedding_weights,
            dense_tensor_ids,
            max_norm=self.max_norm)

        dropout = (self.dropout_keep_probabilities
                   if self.mode == model_fn_lib.ModeKeys.TRAIN
                   else None)

        sequence_lengths =  self._sequence_lengths(sparse_ids)
        if self.bidirectional_rnn:
            cell_fw = rnn_common.construct_rnn_cell(self.num_units, self.cell_type, dropout)
            cell_bw = rnn_common.construct_rnn_cell(self.num_units, self.cell_type, dropout)
            with ops.name_scope('RNN'):
                rnn_outputs, final_states = rnn.bidirectional_dynamic_rnn(cell_fw,
                                                                cell_bw,
                                                                embedding_inputs,
                                                                sequence_length=sequence_lengths,
                                                                dtype=dtypes.float32)
                #outputs = layers.fully_connected(
                #    inputs=array_ops.concat(rnn_outputs, 2),
                #    num_outputs=self.num_units,
                #    activation_fn=self.activation_fn,
                #    trainable=True)
                return array_ops.concat(final_states, 1)
        else:
            cell = rnn_common.construct_rnn_cell(self.num_units, self.cell_type, dropout)
            with ops.name_scope('RNN'):
                rnn_outputs, final_state = rnn.dynamic_rnn(cell,
                                                 embedding_inputs,
                                                 sequence_length=sequence_lengths,
                                                 dtype=dtypes.float32)
        #rnn_outputs = utils.tf_print(rnn_outputs, "rnn_output:")
        #rnn_last_outputs = utils.tf_print(rnn_last_outputs, "rnn_last:")
                #outputs = layers.fully_connected(
                #    inputs=rnn_outputs,
                #    num_outputs=self.num_units,
                #    activation_fn=self.activation_fn,
                #    trainable=True)

                return final_state.h

    def _sequence_lengths(self, sequences):
        line_number = sequences.indices[:,0]
        line_position = sequences.indices[:,1]
        lengths = gen_math_ops.segment_max(data = line_position,
                                           segment_ids = line_number)+1
        lengths = math_ops.cast(lengths, dtypes.int64)
        return lengths

def rnn_column(sparse_id_column,
               embedding_dimension,
               max_sequence_length,
               initializer=None,
               num_units=128,
               activation_fn=None,
               cell_type="basic_rnn",
               bidirectional_rnn=False,
               dropout_keep_probabilities=None,
               mode=model_fn_lib.ModeKeys.TRAIN,
               ckpt_to_load_from=None,
               tensor_name_in_ckpt=None,
               shared_embedding_name=None,
               shared_vocab_size=None,
               max_norm=None,
               trainable=True):
    """Create a `_RNNColumn` for feeding sparse data into a DNN.
    
    Arg:
        sparse_id_column: A `_SparseColumn` which is created by for example
            `sparse_column_with_*` or crossed_column functions. Note that `combiner`
            defined in `sparse_id_column` is ignored.
        embedding_dimension: An integer specifying dimension of the embedding.
        max_sequence_length: max sequence length
        
    """
    if (embedding_dimension is None) or (embedding_dimension < 1):
        raise ValueError('Invalid dimension {}.'.format(dimension))
    if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
        raise ValueError('Must specify both `ckpt_to_load_from` and '
                         '`tensor_name_in_ckpt` or none of them.')

    if (initializer is not None) and (not callable(initializer)):
        raise ValueError('initializer must be callable if specified. '
                         'Embedding of column_name: {}'.format(
                             categorical_column.name))
    if initializer is None:
        initializer = init_ops.truncated_normal_initializer(
            mean=0.0, stddev=1 / math.sqrt(embedding_dimension))

    return _RNNColumn(sparse_id_column,
                      embedding_dimension,
                      max_sequence_length,
                      initializer=initializer,
                      num_units=num_units,
                      activation_fn=activation_fn,
                      cell_type=cell_type,
                      bidirectional_rnn=bidirectional_rnn,
                      dropout_keep_probabilities=dropout_keep_probabilities,
                      mode=mode,
                      ckpt_to_load_from=ckpt_to_load_from,
                      tensor_name_in_ckpt=tensor_name_in_ckpt,
                      shared_embedding_name=shared_embedding_name,
                      shared_vocab_size=shared_vocab_size,
                      max_norm=max_norm,
                      trainable=trainable)
                      

