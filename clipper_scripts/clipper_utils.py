# from https://github.com/ucbrise/clipper/blob/develop/integration-tests/test_utils.py

from __future__ import absolute_import, division, print_function
import os
import pprint
import random
import socket
import docker
import time
import logging

from clipper_admin.container_manager import CLIPPER_DOCKER_LABEL
from clipper_admin import (ClipperConnection, DockerContainerManager,
                           KubernetesContainerManager, CLIPPER_TEMP_DIR,
                           ClipperException)
from clipper_admin import __version__ as clipper_version


# range of ports where available ports can be found
PORT_RANGE = [34256, 40000]

logger = logging.getLogger(__name__)


def log_clipper_state(cl):
    pp = pprint.PrettyPrinter(indent=4)
    logger.info("\nAPPLICATIONS:\n{app_str}".format(app_str=pp.pformat(
        cl.get_all_apps(verbose=True))))
    logger.info("\nMODELS:\n{model_str}".format(model_str=pp.pformat(
        cl.get_all_models(verbose=True))))
    logger.info("\nCONTAINERS:\n{cont_str}".format(cont_str=pp.pformat(
        cl.get_all_model_replicas(verbose=True))))


def get_docker_client():
    if "DOCKER_API_VERSION" in os.environ:
        return docker.from_env(version=os.environ["DOCKER_API_VERSION"])
    else:
        return docker.from_env()


def find_unbound_port():
    """
    Returns an unbound port number on 127.0.0.1.
    """
    while True:
        port = random.randint(*PORT_RANGE)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except socket.error:
            logger.debug(
                "randomly generated port %d is bound. Trying again." % port)


def create_docker_connection(cleanup=True, start_clipper=True):
    logger.info("Creating DockerContainerManager")
    cm = DockerContainerManager(
        clipper_query_port=find_unbound_port(),
        clipper_management_port=find_unbound_port(),
        clipper_rpc_port=find_unbound_port(),
        redis_port=find_unbound_port())
    cl = ClipperConnection(cm)
    if cleanup:
        cl.stop_all()
        docker_client = get_docker_client()
        docker_client.containers.prune(filters={"label": CLIPPER_DOCKER_LABEL})
    if start_clipper:
        # Try to start Clipper in a retry loop here to address flaky tests
        # as described in https://github.com/ucbrise/clipper/issues/352
        while True:
            try:
                logger.info("Starting Clipper")
                cl.start_clipper()
                time.sleep(1)
                break
            except docker.errors.APIError as e:
                logger.info(
                    "Problem starting Clipper: {}\nTrying again.".format(e))
                cl.stop_all()
                cm = DockerContainerManager(
                    clipper_query_port=find_unbound_port(),
                    clipper_management_port=find_unbound_port(),
                    clipper_rpc_port=find_unbound_port(),
                    redis_port=find_unbound_port())
                cl = ClipperConnection(cm)
    else:
        cl.connect()
    return cl


def create_kubernetes_connection(cleanup=True,
                                 start_clipper=True,
                                 connect=True,
                                 with_proxy=False,
                                 kubernetes_proxy_addr="127.0.0.1:8080",
                                 num_frontend_replicas=1):
    logger.info("Creating KubernetesContainerManager")
    if with_proxy:
        cm = KubernetesContainerManager(kubernetes_proxy_addr=kubernetes_proxy_addr)
    else:
        cm = KubernetesContainerManager()
    cl = ClipperConnection(cm)
    if cleanup:
        cl.stop_all()
        # Give kubernetes some time to clean up
        time.sleep(20)
        logger.info("Done cleaning up clipper")
    if start_clipper:
        logger.info("Starting Clipper")
        cl.start_clipper(
            query_frontend_image=
            "568959175238.dkr.ecr.us-west-1.amazonaws.com/clipper/query_frontend:{}".
            format(clipper_version),
            mgmt_frontend_image=
            "568959175238.dkr.ecr.us-west-1.amazonaws.com/clipper/management_frontend:{}".
            format(clipper_version),
            num_frontend_replicas=num_frontend_replicas)
        time.sleep(1)
    if connect:
        try:
            cl.connect()
        except Exception:
            pass
        except ClipperException:
            pass
        return cl


def log_docker(clipper_conn):
    """Retrieve status and log for last ten containers"""
    container_runing = clipper_conn.cm.docker_client.containers.list(limit=10)
    logger.info('----------------------')
    logger.info('Last ten containers status')
    for cont in container_runing:
        logger.info('Name {}, Image {}, Status {}, Label {}'.format(
            cont.name, cont.image, cont.status, cont.labels))

    logger.info('----------------------')
    logger.info('Printing out logs')

    for cont in container_runing:
        logger.info('Name {}, Image {}, Status {}, Label {}'.format(
            cont.name, cont.image, cont.status, cont.labels))
        logger.info(cont.logs())

