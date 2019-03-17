import utils
import numpy as np
import collections

from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions.laplace import Laplace
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops.losses import losses

from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.summary import summary
from tensorflow.python.ops.losses import util


LossSpec = collections.namedtuple(
    'LossSpec', ['training_loss', 'unreduced_loss', 'processed_labels'])


class _RegressionHeadWithLaplaceNLL(head_lib._Head):
    """`Head` for regression using the negative log-likelihood of a Laplace distribution."""
    def __init__(self,
                 m = 1,
                 logits_dimension=1,
                 loss_reduction=losses.Reduction.SUM,
                 name=None):

        """`Head` for regression."""
        self._name = name
        self._logits_dimension = logits_dimension
        self._loss_reduction = loss_reduction
        self._m = m

    @property
    def name(self):
        return self._name
    
    @property
    def logits_dimension(self):
        return self._logits_dimension

    def create_loss(self, features, mode, mus, sigmas, alphas, labels):
        """See `Head`."""

        labels = array_ops.expand_dims(labels, 1)
        #unweighted_loss = self._mean_log_gaussian_like(
        #unweighted_loss = self._mean_log_laplace_like(
        unweighted_loss = self._laplace_nll(
            labels=labels, mus=mus, sigmas=sigmas, alphas=alphas)
        #unweighted_loss = _laplace_nll(
        #    labels=labels, mus=mus, sigmas=sigmas, alphas=alphas)
        training_loss = losses.compute_weighted_loss(
            unweighted_loss, reduction=self._loss_reduction)

        return LossSpec(
            training_loss=training_loss,
            unreduced_loss=unweighted_loss,
            processed_labels=labels)

    def create_loss2(self, features, mode, logits, labels):
        """See `Head`."""
        del mode  # Unused for this head.
        print(labels)
        labels = head_lib._check_dense_labels_match_logits_and_reshape(
            labels=labels, logits=logits, expected_labels_dimension=1)

        labels = math_ops.to_float(labels)
        unweighted_loss = losses.mean_squared_error(
            labels=labels,
            predictions=logits,
            reduction=losses.Reduction.NONE)
        training_loss = losses.compute_weighted_loss(
            unweighted_loss, reduction=self._loss_reduction)

        return LossSpec(
            training_loss=training_loss,
            unreduced_loss=unweighted_loss,
            processed_labels=labels)

    def create_estimator_spec(
            self, features, logits, mode, labels=None, train_op_fn=None):
        """See `Head`."""

        # split logits into mu, sigma and alpha
        components = array_ops.reshape(logits, [-1, 3, self._m])
        mus = components[:, 0, :]
        sigmas = components[:, 1, :]
        alphas = components[:, 2, :]
        alphas = nn_ops.softmax(clip_ops.clip_by_value(alphas, 1e-2, 1.))

        # Predict.
        with ops.name_scope('head'):
            #logits = head_lib._check_logits(logits, self._logits_dimension)
            means = math_ops.reduce_sum(alphas*mus, axis=1, keepdims=True)

            uncertainty = math_ops.reduce_sum(
                alphas*sigmas, axis=1, keepdims=True)
            
            predicted_value = array_ops.concat([means, uncertainty], 1)
            predictions = {prediction_keys.PredictionKeys.PREDICTIONS:
                           predicted_value}
            if mode == model_fn.ModeKeys.PREDICT:
                regression_output = export_output.RegressionOutput(
                    value=predicted_value)
                return model_fn.EstimatorSpec(
                    mode=model_fn.ModeKeys.PREDICT,
                    predictions=predictions,
                    export_outputs={
                        head_lib._DEFAULT_SERVING_KEY: regression_output,
                        head_lib._REGRESS_SERVING_KEY: regression_output,
                        head_lib._PREDICT_SERVING_KEY:
                        export_output.PredictOutput(predictions)
                    })
            
            # Eval.
            if mode == model_fn.ModeKeys.EVAL:
                # Estimator already adds a metric for loss.
                mus = math_ops.reduce_sum(alphas*mus, axis=1, keepdims=True)
                #mus = utils.tf_print(mus, "mus:")
                #labels = utils.tf_print(labels, "labels:")
                training_loss, unweighted_loss, _ = self.create_loss2(
                    features=features, mode=mode, logits=mus, labels=labels)
                keys = metric_keys.MetricKeys

                eval_metric_ops = {
                    head_lib._summary_key(self._name, 
                        keys.LOSS_MEAN) : 
                            metrics_lib.mean(
                                unweighted_loss, weights=None)
                }
                return model_fn.EstimatorSpec(
                    mode=model_fn.ModeKeys.EVAL,
                    predictions=predictions,
                    loss=training_loss,
                    eval_metric_ops=eval_metric_ops)

            # Train.
            if train_op_fn is None:
                raise ValueError('train_op_fn can not be None.')

            training_loss, unweighted_loss, _ = self.create_loss(
                features=features, mode=mode, mus=mus,
                sigmas=sigmas, alphas=alphas, labels=labels)

        with ops.name_scope(''):
            summary.scalar(
                head_lib._summary_key(self._name,
                                      metric_keys.MetricKeys.LOSS_MEAN),
                losses.compute_weighted_loss(
                    unweighted_loss,
                    reduction=losses.Reduction.MEAN))
            return model_fn.EstimatorSpec(
                mode=model_fn.ModeKeys.TRAIN,
                predictions=predictions,
                loss=training_loss,
                train_op=train_op_fn(training_loss))

    def _log_sum_exp(self, x, axis=None):
        """Log-sum-exp trick implementation"""
        x_max = math_ops.reduce_max(x, axis=axis, keepdims=True)
        return math_ops.log(math_ops.reduce_sum(math_ops.exp(x - x_max),
                            axis=axis, keepdims=True))+x_max


    def _mean_log_gaussian_like(self, labels, mus, sigmas, alphas):
        exponent = math_ops.log(alphas) - .5 * float(self._c) *  math_ops.log(2 * np.pi) \
        - float(self._c) * math_ops.log(sigmas) \
        - math_ops.reduce_sum((labels - mus)**2, axis=1)/(2*(sigmas)**2)

        log_gauss = self._log_sum_exp(exponent, axis=1)
        res = -log_gauss
        return res

    def _mean_log_laplace_like(self, labels, mus, sigmas, alphas):
        exponent = math_ops.log(alphas) - float(self._c) *  math_ops.log(2 * sigmas) \
        - math_ops.reduce_sum(math_ops.abs(labels - mus), axis=1)/sigmas

        log_gauss = self._log_sum_exp(exponent, axis=1)
        res = -log_gauss
        return res

    def _laplace_nll(self, labels, mus, sigmas, alphas, scope=None,
                 loss_collection=ops.GraphKeys.LOSSES):
        with ops.name_scope(scope, "log_Laplace_like",
                        (mus, sigmas, alphas)) as scope:
            eps = 1e-6
        
            dist = Normal(loc=mus, scale=sigmas)

            #labels = utils.tf_print(labels, "label!!:")
            loss = - math_ops.log(math_ops.reduce_sum(
                alphas * dist.prob(labels), axis=1) + eps)
            util.add_loss(loss, loss_collection)
            return loss
  

def _regression_head_with_laplace_nll(m=1,
                                      logits_dimension=1,
                                      name=None):
    return _RegressionHeadWithLaplaceNLL(m=m,
                                         logits_dimension=logits_dimension,
                                         name=name)

