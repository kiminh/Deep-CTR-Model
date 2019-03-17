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
import sentencepiece as spm

from collections import defaultdict
from itertools import chain, combinations
from tensorflow.contrib.learn.python.learn import run_config


class BpeModel:
    def __init__(self, model_file):
        self._model = spm.SentencePieceProcessor()
        if not self._model.Load(model_file):
            raise ValueError("bpe model loading error")
        self._eos_id = 2

    def encode_as_pieces(self, sentence, max_len):
        pieces = self._model.Encode(sentence)
        if len(pieces) > max_len:
            return pieces[:max_len]
        else:
            return (pieces + ['']*(max_len-len(pieces)))

    def encode_as_ids(self, sentence, max_len):
        ids = self._model.EncodeAsIds(sentence)
        if len(ids) > max_len:
            return ids[:max_len]
        else:
            return (ids + [self._eos_id]*(max_len-len(ids)))


def load_bpe_model(bpe_model_file):
    try:
        bpe_model = BpeModel(bpe_model_file)
        return bpe_model
    except:
        bpe = None
    return bpe


def tf_print(tensor, message='', sparse_tensor=False):
    """
    print a tensor for debugging
    """
    def _print_tensor(tensor):
        print(message, tensor)
        return tensor

    if not sparse_tensor:
        log_op = tf.py_func(_print_tensor, [tensor], [tensor.dtype])[0]
        with tf.control_dependencies([log_op]):
            res = tf.identity(tensor)

        return res
    else:
        tensor_values = tensor.values
        log_op = tf.py_func(_print_tensor, [tensor_values], [tensor_values.dtype])[0]
        with tf.control_dependencies([log_op]):
            res = tf.identity(tensor_values)

        return res


def gizp_reader_fn():
    """
        gzip reader function
    """
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
                             compression_type=tf.python_io.TFRecordCompressionType.GZIP))
                    
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def tf_confidence(mu, impressions):
    n = impressions
    if n == 0: return 0
    z = 1.96 #1.96 -> 95% confidence
    phat = tf.abs(mu)
    denorm = 1. + (z*z/n)
    enum1 = phat + z*z/(2*n)
    enum2 = z * tf.sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    return (enum1+enum2)/denorm


def tf_confidence2(mu, impressions):
    confidence =  1.96*tf.sqrt((mu * (1. - mu))/impressions)
    return confidence


def list_flatten(l):
    return reduce(lambda x,y: x.extend(y) or x, l)


def get_line_count(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines


# ad stat info class
class AdStatInfo:
    def __init__(self, ad_stat_info_file):
        self.pos_imp_dict = defaultdict(lambda: 0)
        self.pos_clk_dict = defaultdict(lambda: 0)
        self.ad_pos_count_dict = {}
        self.ad_clk_dict = defaultdict(lambda: 0)
        self.ad_imp_dict = defaultdict(lambda: 0)
        self.ad_coec_dict = defaultdict(lambda: 0.0)
        self.loaded = False
        if ad_stat_info_file:
            self._load_ad_stat_info(ad_stat_info_file)
            self.loaded = True
    
    # load ad stat info file
    #    line format: ad_id,keyword,position,click,impression
    def _load_ad_stat_info(self, ad_stat_info_file):
        # line parser
        def _ad_stat_info_line_parse(line):
            try :
                p_line = line.split(',')
                ad = p_line[0].strip()
                keyword = p_line[1].strip()
                imp = int(p_line[4].strip())
                clk = int(p_line[3].strip())
                pos = int(p_line[2].strip())
                return ad, keyword, pos, clk, imp
            except :
                raise ValueError("line parsing errors: {}".format(p_line))

        with open(ad_stat_info_file) as ad_stat_info_f:
            for stat_line in ad_stat_info_f:
                ad, keyword, pos, clk, imp = _ad_stat_info_line_parse(stat_line)
                ad_key = ad + keyword

                # pos counter,
                self.pos_imp_dict[pos] += imp
                self.pos_clk_dict[pos] += clk
                self.ad_imp_dict[ad_key] += imp
                self.ad_clk_dict[ad_key] += clk
                if ad_key not in self.ad_pos_count_dict:
                    self.ad_pos_count_dict[ad_key] = {}
                if pos not in self.ad_pos_count_dict[ad_key]:
                    self.ad_pos_count_dict[ad_key][pos] = [0,0]
                self.ad_pos_count_dict[ad_key][pos][0] += clk
                self.ad_pos_count_dict[ad_key][pos][1] += imp

            for ad_key, pos_count_dict in self.ad_pos_count_dict.items():
                pos_ec = \
                         sum([clk_imp[1]*float(self.pos_clk_dict[pos])/float(self.pos_imp_dict[pos]) for pos, clk_imp in pos_count_dict.items()])
                self.ad_coec_dict[ad_key] = self.ad_clk_dict[ad_key]/pos_ec

    def get_coec(self, ad_key):
        return self.ad_coec_dict[ad_key]
    
    def get_click(self, ad_key):
        return self.ad_clk_dict[ad_key]

    def get_impression(self, ad_key):
        return self.ad_imp_dict[ad_key]

    def get_pos_bias(self, pos):
        if self.pos_imp_dict[pos] == 0:
            return 0.0
        return float(self.pos_clk_dict[pos])/self.pos_imp_dict[pos]
