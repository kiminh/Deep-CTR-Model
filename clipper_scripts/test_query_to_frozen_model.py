import tensorflow as tf
import argparse
import sentencepiece as spm
import codecs

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.core.framework import graph_pb2
from tensorflow.python.saved_model import loader

"""
["ad_id", [""]],
        ["total_rank", [0]],
        ["ad_headline", [""]],
        ["ad_desc", [""]],
        ["ad_imp", [0.0]]
"""
_EOS_ID = 2

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _make_ad_feacture_dict(ad_id,
                           total_rank,
                           ad_headline):
    feature_dict = {
        "ad_id": _bytes_feature(ad_id),
        "total_rank": _int64_feature(total_rank),
        "ad_headline": _int64list_feature(ad_headline)
    }
    return feature_dict


def get_examples(sp, query_file, max_len):
    result = []
    for s in codecs.open(query_file, 'r', encoding='utf8'):
        arr = s.split('\t')
        ad_id = str(arr[0].decode('ascii'))
        total_rank = int(arr[1])
        ad_imp = float(arr[3])
        ad_headline = sp.EncodeAsIds(arr[2])
        if len(ad_headline) > max_len:
            ad_headline = ad_headline[:max_len]
        else:
            ad_headline = ad_headline + [_EOS_ID]*(max_len - len(ad_headline))

        # replace 0 with 1 for utf-8 decoding problem
        ad_headline = map(lambda x: 1 if x == 0 else x, ad_headline)
        print(ad_id, total_rank, ad_headline)
        ad_feature_dict = _make_ad_feacture_dict(
            ad_id, total_rank, ad_headline)
        example = tf.train.Example(
            features=tf.train.Features(feature=ad_feature_dict))
        serialized_example = example.SerializeToString()
        result.append(serialized_example)
    return result


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir",
                        default="../model_dir/",
                        type=str, help="Frozen model file to import")
    parser.add_argument("--bpe_model", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=15)

    args = parser.parse_args()

    if args.bpe_model:
        sp = spm.SentencePieceProcessor()
        if not sp.Load(args.bpe_model):
            raise ValueError("loading bpe model error")

    examples = get_examples(sp, args.query_file, args.max_len)
    example_lens = len(examples)
    print(example_lens)

    DEFAULT_TAGS = 'serve'
    tags = DEFAULT_TAGS
    with tf.Graph().as_default() as graph:
        session = tf.Session(graph=graph)
        loader.load(session, tags.split(","), args.export_dir)
        pred = session.run('ddn/input_from_feature_columns/output_layer/Sigmoid:0', feed_dict={'input_example_tensor:0': examples})
        print(pred)
        #for op in graph.get_operations():
        #    print(op)
 
