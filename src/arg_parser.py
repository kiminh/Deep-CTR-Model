from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import argparse

def create_arg_parser(mode=tf.estimator.ModeKeys.TRAIN):
    """
    parse command arguments
    """
    parser = argparse.ArgumentParser()

    if mode == tf.estimator.ModeKeys.TRAIN:
        parser.add_argument(
            '--train_filename',
            type=str,
            required=True)
        parser.add_argument(
            '--eval_filename',
            type=str,
            required=True)
        parser.add_argument(
            '--output_dir',
            type=str,
            default=None)
        parser.add_argument(
        '--train_steps',
        default=None,
        type=str)
    else: # eval or predict
        parser.add_argument(
            '--data_file_list',
            type=str,
            action='append',
            required=True)
        parser.add_argument(
            '--model_dir',
            type=str,
            required=True)
        parser.add_argument(
            '--output_file',
            type=str,
            required=True)
        parser.add_argument(
        '--infer_steps',
        default=None,
        type=str)

    parser.add_argument(
        '--column_config_file',
        type=str,
        required=True)
    parser.add_argument(
        '--model_config_file',
        type=str,
        required=True)
    parser.add_argument(
        '--model_type',
        type=str,
        default="")
    parser.add_argument(
        '--bpe_model_path',
        type=str,
        default=None)
    parser.add_argument(
       '--enable_hvd',
       action='store_true',
       default=False)
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='num of epoch',
        default=1)
    parser.add_argument(
        '--batch_size',
        default=512,
        type=int)
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--optimizer',
        default='Adam',
        type=str)
    return parser
