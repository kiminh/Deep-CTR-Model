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

#
# train and eval func
#
def train_and_eval(model_type, train_filename,  eval_filename,
                   column_config_file, model_config_file, bpe_model_path,
                   output_dir, train_steps, num_epochs, batch_size,
                   enable_hvd, optimizer, learning_rate, **hparams):

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    run_config = tf.contrib.learn.RunConfig(
        session_config=sess_config)

    hooks = []
    if enable_hvd:
        import horovod.tensorflow as hvd
        hvd.init()

        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        hooks.append(bcast_hook)
        if train_steps == None:
            raise ValueError("steps should be defined")
        else:
            train_steps = train_steps//hvd.size()

        train_filename = "%s.%d" % (train_filename, hvd.rank())
        eval_filename  = "%s.%d" % (eval_filename, hvd.rank())
    else:
        hvd = None
     
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
   
    if hvd: 
        model_dir = output_dir if hvd.rank() == 0 else None
    else:
        model_dir = output_dir

    # loading model config
    model_config = load_model_config(model_type, model_config_file)

    # estimator
    custom_estimator = get_custom_estimator(model_type, model_config, column_config_file,
        optimizer, learning_rate, model_dir, hvd, run_config)

    tf_estimator = custom_estimator.get_tf_estimator()

    column_config = custom_estimator.get_column_config()

    # train dataset
    train_dataset = Dataset(train_filename, batch_size, num_epochs, column_config)

    # train
    tf_estimator.train(input_fn=train_dataset.input_fn, steps=train_steps, hooks=hooks)

    # eval dataset
    eval_dataset = Dataset(eval_filename, batch_size, 1, column_config, mode="eval")

    # evaluate
    tf_estimator.evaluate(input_fn=eval_dataset.input_fn)
   

def main(unused_argv):
    args = arg_parser.create_arg_parser().parse_args()
    train_and_eval(**vars(args))


if __name__ == "__main__":
    tf.app.run()
