from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tempfile
import os
import utils
import arg_parser

from custom_estimator import get_custom_estimator
from dataset import Dataset
from json_config_parser import load_model_config


tf.logging.set_verbosity(tf.logging.INFO)


def predict(model_type, data_file_list, output_file,
            column_config_file, model_config_file, bpe_model_path,
            model_dir, infer_steps, num_epochs, batch_size,
            enable_hvd, optimizer, learning_rate, **hparams):

    hooks = []
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    run_config = tf.contrib.learn.RunConfig(
        session_config=sess_config)

    if enable_hvd:
        import horovod.tensorflow as hvd
        hvd.init()

        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        hooks.append(bcast_hook)
        if steps == None:
            raise ValueError("steps should be defined")
        else:
            steps = steps//hvd.size()
        
        data_file_list = [ "%s.%d" % (filename, hvd.rank()) for filename in data_file_list]
    else:
        hvd = None
     
    # loading model config
    model_config = load_model_config(model_type, model_config_file)

    # loading a bpe model
    #  bpe_model_obj = utils.load_bpe_model(bpe_model)

    # Estimator
    custom_estimator = get_custom_estimator(model_type, model_config, column_config_file,
        optimizer, learning_rate, model_dir, hvd, run_config)
   
    tf_estimator = custom_estimator.get_tf_estimator()

    # input_fn
    dataset = Dataset(data_file_list,
                      batch_size,
                      1,
                      custom_estimator.get_column_config(),
                      mode="predict")

    ctr_prob_results = tf_estimator.predict(input_fn=dataset.input_fn, 
        predict_keys=["probabilities"], hooks=hooks)

    # extract click through rate
    ctr_prob_results = map(lambda x:str(x["probabilities"][1]), ctr_prob_results)

    if hvd != None:
        output_file = "{}.{}".format(output_file, hvd.rank())

    with open(output_file, "w") as output_f:
        output_f.write("{}".format('\n'.join(ctr_prob_results)))

def main(unused_argv):
    args = arg_parser.create_arg_parser("predict").parse_args()
    predict(**vars(args))


if __name__ == "__main__":
    tf.app.run()
