# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import sys
import os
import time
import argparse
import clipper_utils

from clipper_admin.deployers.tensorflow import deploy_tensorflow_model, create_endpoint


input_tensor_name = 'input_example_tensor:0'
output_tensor_name = 'ddn/input_from_feature_columns/output_layer/Sigmoid:0'


def _predict(sess, inp):
    preds = sess.run(output_tensor_name,
                     feed_dict={input_tensor_name: inp})
    preds = [str(p[0]) for p in preds]
    return preds


def deploy_model(clipper_conn,
                 model_name,
                 app_name,
                 sess,
                 version,
                 input_type,
                 link_model=False,
                 predict_fn=_predict):
    deploy_tensorflow_model(clipper_conn, model_name, version, input_type,
                            predict_fn, sess)
    time.sleep(5)
    
    if link_model:
        clipper_conn.link_model_to_app(app_name=app_name,
                                       model_name=model_name)
        time.sleep(5)

                
def _get_lastest_model_version(model_path):
    entries = [f for f in os.listdir(model_path) if not os.path.isfile(f)]
    entries = map(lambda x: int(x), entries)
    entries.sort()
 
    return entries[-1]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_export_dir",
                        type=str,
                        required=True)
    parser.add_argument("--app_name",
                        type=str,
                        required=True)
    parser.add_argument("--model_name",
                        type=str,
                        required=True)
    args = parser.parse_args()

    clipper_conn = clipper_utils.create_docker_connectdion(
        cleanup=True, start_clipper=True)
    clipper_utils.log_clipper_state(clipper_conn)

    #register 
    clipper_conn.register_application(args.app_name,
                                      "strings",
                                      "0.5",
                                      100000000)
    time.sleep(1)

    export_dir = args.model_export_dir
    version = _get_lastest_model_version(export_dir)

    #cd model_dir
    os.chdir(export_dir)
    current_dir = os.getcwd()
    print(current_dir)

    sess = str(version)
    print(sess)

    # deploy model
    deploy_model(clipper_conn,
                 args.model_name,
                 args.app_name,
                 sess,
                 version,
                 "strings",
                 link_model=True)

    # check 
    model = clipper_conn.get_linked_models(app_name=args.app_name)

    num = clipper_conn.cm.get_num_replicas(name=model[0],
                                           version=version)
    print(num)
    if num == 0:
        print("model container crashed!")
        #clipper_conn.get_clipper_logs()

    print(clipper_conn.inspect_instance())
