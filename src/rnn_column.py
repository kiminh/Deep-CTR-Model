import math
import collections
import utils

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import embedding_ops
from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers.feature_column import _FeatureColumn
from tensorflow.contrib.layers.python.layers.feature_column import _get_feature_config 
from tensorflow.contrib.layers.python.layers.feature_column import _is_variable
from tensorflow.contrib.layers.python.layers.feature_column import _maybe_restore_from_checkpoint 
from tensorflow.contrib.layers.python.layers.feature_column import _embeddings_from_arguments
from tensorflow.contrib.layers.python.layers.feature_column import _DeepEmbeddingLookupArguments
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.platform import tf_logging as logging

class _RNNColumn(
    _FeatureColumn,
    fc_core._DenseColumn,
    collections.namedtuple("_RNNColumn", [
        "sparse_id_column", "embedding_dimension", "max_sequence_length",
        "initializer", "num_units", "cell_type", 
        "bidirectional_rnn", "mode", "dropout_keep_probabilities",
        "ckpt_to_load_from", "tensor_name_in_ckpt", "shared_embedding_name",
        "shared_vocab_size", "max_norm", "trainable"])):

    def __new__(cls,
                sparse_id_column,
                embedding_dimension,
                max_sequence_length,
                initializer=None,
                num_units=256,
                cell_type='basic_rnn',
                bidirectional_rnn=False,
                mode=model_fn.ModeKeys.TRAIN,
                dropout_keep_probabilities=None,
                ckpt_to_load_from=None,
                tensor_name_in_ckpt=None,
                shared_embedding_name=None,
                shared_vocab_size=None,
                max_norm=None,
                trainable=True):
        if initializer is not None and not callable(initializer):
            raise ValueError("initializer must be callable if specified. "
                             "Embedding of column_name: {}".format(
                                 sparse_id_column.name))

        if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
            raise ValueError("Must specify both `ckpt_to_load_from` and "
                             "`tensor_name_in_ckpt` or none of them.")        
 
        if initializer is None:
            logging.warn("The default stddev value of initializer will change from "
                         "\"1/sqrt(vocab_size)\" to \"1/sqrt(dimension)\" after "
                         "2017/02/25.")
            stddev = 1 / math.sqrt(sparse_id_column.length)
            initializer = init_ops.truncated_normal_initializer(
                mean=0.0, stddev=stddev)

        return super(_RNNColumn, cls).__new__(cls, 
                                              sparse_id_column,
                                              embedding_dimension,
                                              max_sequence_length,
                                              initializer,
                                              num_units, 
                                              cell_type,
                                              bidirectional_rnn,
                                              mode,
                                              dropout_keep_probabilities,
                                              ckpt_to_load_from,
                                              tensor_name_in_ckpt,
                                              shared_embedding_name,
                                              shared_vocab_size,
                                              max_norm,
                                              trainable)

    @property
    def name(self):
        return "{}_rnn".format(self.sparse_id_column.name)
       
    @property
    def length(self):
        """Returns vocabulary or hash_bucket size."""
        if self.shared_vocab_size is None:
            return self.sparse_id_column.length
        else:
            return self.shared_vocab_size

    @property
    def config(self):
        return _get_feature_config(self.sparse_id_column)

    @property
    def key(self):
        """Returns a string which will be used as a key when we do sorting."""
        #print("key:{}".format(self))
        return "{}".format(self)

    def insert_transformed_feature(self, columns_to_tensors):
        if self.sparse_id_column not in columns_to_tensors:
            self.sparse_id_column.insert_transformed_feature(columns_to_tensors)
        columns_to_tensors[self] = columns_to_tensors[self.sparse_id_column]


    def _to_dnn_input_layer(self,
                            transformed_input_tensor,
                            weight_collections=None,
                            trainable=True,
                            output_rank=2):
        """Returns a Tensor as an input to the first layer of neural network.
        Args:
            transformed_input_tensor: A tensor that has undergone the transformations
            in `insert_transformed_feature`. Rank should be >= `output_rank`.
            unused_weight_collections: Unused. One hot encodings are not variable.
            unused_trainable: Unused. One hot encodings are not trainable.
            output_rank: the desired rank of the output `Tensor`.

        Returns:
            A outputs Tensor of RNN to be fed into the first layer of neural network.

        Raises:
        """
        sparse_id_column = self.sparse_id_column.id_tensor(transformed_input_tensor)
        # pylint: disable=protected-access
        sparse_id_column = layers._inner_flatten(sparse_id_column, output_rank)

        batch_size = sparse_id_column.dense_shape[0]
        dense_id_tensor = sparse_ops.sparse_to_dense(sparse_id_column.indices,
                                                     [batch_size, 
                                                      self.max_sequence_length],
                                                     sparse_id_column.values,
                                                     default_value=0)
       # dense_id_tensor = gen_array_ops.reshape(dense_id_tensor, [-1, self.max_sequence_length])

        if self.shared_embedding_name is not None:
            shared_embedding_collection_name = (
                "SHARED_EMBEDDING_COLLECTION_" + self.shared_embedding_name.upper())
            graph = ops.get_default_graph()
            shared_embedding_collection = (
                graph.get_collection_ref(shared_embedding_collection_name))
            shape = [self.length, self.embedding_dimension]
            if shared_embedding_collection:
                if len(shared_embedding_collection) > 1:
                    raise ValueError(
                        "Collection %s can only contain one "
                        "(partitioned) variable." % shared_embedding_collection_name)
                else:
                    embeddings = shared_embedding_collection[0]
                    if embeddings.get_shape() != shape:
                        raise ValueError(
                            "The embedding variable with name {} already "
                            "exists, but its shape does not match required "
                            "embedding shape here. Please make sure to use "
                            "different shared_embedding_name for different "
                            "shared embeddings.".format(args.shared_embedding_name))
            else:
                embeddings = contrib_variables.model_variable(
                    name=self.shared_embedding_name,
                    shape=shape,
                    dtype=dtypes.float32,
                    initializer=self.initializer,
                    trainable=(trainable and self.trainable),
                    collections=weight_collections)
                graph.add_to_collection(shared_embedding_collection_name, embeddings)
        else:
            embeddings = contrib_variables.model_variable(
                name="weights",
                shape=[self.length, self.embedding_dimension],
                dtype=dtypes.float32,
                initializer=self.initializer,
                trainable=(trainable and self.trainable),
                collections=weight_collections)

        if _is_variable(embeddings):
            embeddings = [embeddings]
        else:
            embeddings = embeddings._get_variable_list()  # pylint: disable=protected-access
       
        embedding_inputs = embedding_lookup(
            embeddings,
            dense_id_tensor,
            max_norm=self.max_norm)
        
        dropout = (self.dropout_keep_probabilities
                   if self.mode == model_fn.ModeKeys.TRAIN
                   else None)

        sequence_length =  self._sequence_length(dense_id_tensor)
        if bidirectional_rnn:
            cell_fw = rnn_common.construct_rnn_cell(self.num_units, self.cell_type, dropout)
            cell_bw = rnn_common.construct_rnn_cell(self.num_units, self.cell_type, dropout)
            _rnn_outputs, _ = rnn.bidirectional_dynamic_rnn(cell_fw,
                                                            cell_bw,
                                                            embedding_inputs,
                                                            sequence_length=sequence_length,
                                                            dtype=dtypes.float32)
            rnn_outputs = array_ops.concat(_rnn_outputs, axis=2)
        else:
            cell = rnn_common.construct_rnn_cell(self.num_units, self.cell_type, dropout)
            rnn_outputs, _ = rnn.dynamic_rnn(cell,
                                             embedding_inputs,
                                             sequence_length=sequence_length,
                                             dtype=dtypes.float32)
        
        return self._extract_last_relevent(rnn_outputs, sequence_length)

    def _extract_last_relevent(self, output, length):
        batch_size = array_ops.shape(output)[0]
        max_sequence_length = int(output.get_shape()[1])
        num_units = int(output.get_shape()[2])
        index = math_ops.range(0, batch_size) * max_sequence_length + (length - 1)
        flat = array_ops.reshape(output, [-1, num_units])
        relevant = array_ops.gather(flat, index)
        return relevant

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        return self._to_dnn_input_layer(inputs.get(self), weight_collections, trainable)

    def _checkpoint_path(self):
        if self.ckpt_to_load_from is not None:
            return self.ckpt_to_load_from, self.tensor_name_in_ckpt
        return None

    def _sequence_length(self, sequences):
        used = math_ops.sign(sequences)
        length = math_ops.reduce_sum(used, 1)
        length = math_ops.cast(length, dtypes.int32)
        return length

    @property
    def _parse_example_spec(self):
        return self.config

def rnn_column(sparse_id_column,
               embedding_dimension,
               max_sequence_length,
               initializer=None,
               num_units=128,
               cell_type="basic_rnn",
               bidirectional_rnn=False,
               dropout_keep_probabilities=None,
               mode=model_fn.ModeKeys.TRAIN,
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
    return _RNNColumn(sparse_id_column,
                      embedding_dimension,
                      max_sequence_length,
                      initializer=initializer,
                      num_units=num_units,
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
                      

