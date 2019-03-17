# -*- coding: utf-8 -*-

from __future__ import print_function
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers import python as python_deployer
import json
import requests
from datetime import datetime
import time
import numpy as np
import signal
import sys
import tensorflow as tf
import argparse
import sentencepiece as spm
import codecs
import urllib
import sys


_EOS_ID = 2


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _make_pwc_ad_feacture_dict(ad_id,
                               total_rank,
                               ad_headline_ids):
    feature_dict = {
        "ad_id": _bytes_feature(ad_id),
        "total_rank": _int64_feature(total_rank),
        "ad_headline": _int64list_feature(ad_headline_ids)
    }
    return feature_dict


def truncate_list(l, max_len):
    if len(l) > max_len:
        l = l[:max_len]
    else:
        l = l + [_EOS_ID]*(max_len - len(l))
    return map(lambda x: 1 if x == 0 else x, l[:max_len])


def get_examples(sp, query_file, max_len):
    result = []
    ad_list = []
    with open(query_file, 'r') as f:
        for line in f:
            arr = line.split('\t')
            ad_id = arr[0]
            total_rank = int(arr[1].strip())
            ad_imp = float(arr[3].strip())
            ad_headline = arr[2].strip()
            ad_headline_ids = truncate_list(
                sp.EncodeAsIds(ad_headline),
                max_len)
            ad_feature_dict = _make_pwc_ad_feacture_dict(
                ad_id, total_rank, ad_headline_ids)
            example = tf.train.Example(
                features=tf.train.Features(feature=ad_feature_dict))
            serialized_example = example.SerializeToString()

            ad_list.append([ad_id, total_rank, ad_headline, ad_imp])
            result.append(serialized_example)
    return result, ad_list


def confidence(ctr, impressions):
    n = impressions
    if n == 0:
        return 0
    z = 1.96 #1.96 -> 95% confidence
    phat = ctr
    denorm = 1. + (z*z/n)
    enum1 = phat + z*z/(2*n)
    enum2 = z * np.sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    return (enum1-enum2)/denorm, (enum1+enum2)/denorm


def wilson(ctr, impressions):
    if impressions == 0:
        return 0
    else:
        return confidence(ctr, impressions)


def make_outputs(ad_list, result, batch=False):
    outputs = []
    result = json.loads(result)
    if batch:
        for ad, output in zip(ad_list, result["batch_predictions"]):
            ctr = float(output["output"])
            ad_imp = ad[3]
            if ctr == 0.0:
                variance = 3./ad_imp
            else:
                variance = wilson(ctr, ad_imp)[1]
            outputs.append((ad[0], output["output"], variance, ad[3]))
            print(ad[0], ad[2], ad[3], output["output"], variance)
    else:
        pass

    return outputs


def predict(sp, addr, app_name, query_file, max_len, batch=False):
    url = "http://%s/%s/predict" % (addr, app_name)
 
    x, ad_list = get_examples(sp, query_file, max_len)
    if batch:
        req_json = json.dumps({'input_batch': x}, ensure_ascii=False)
    else:
        req_json = json.dumps({'input': x[9]}, ensure_ascii=False)
        #req_dict = MessageToJson(req_dict)

    headers = {'Content-type': 'application/json'}
    start = datetime.now()
    r = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    outputs = make_outputs(ad_list, r.text, batch)
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % ("OK", latency))
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe_model", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--ad_headline_max_len", type=int, default=15)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    app_name = "pwc-ad-candidate-app"
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.connect()
 
    time.sleep(2)
    
    if args.bpe_model:
        sp = spm.SentencePieceProcessor()
        if not sp.Load(args.bpe_model):
            raise ValueError("loading bpe model error")
        print("bpe model loaded")

    addr = clipper_conn.get_query_addr()
    outputs = predict(
        sp, addr, app_name, args.query_file,
        args.ad_headline_max_len, True)
    with open(args.output_file, "w") as f:
        for output in outputs:
            f.write("{}\t{}\t{}\t{}\n".format(output[0], output[1], output[2], output[3]))
                    
