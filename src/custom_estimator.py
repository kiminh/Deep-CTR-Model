from feature_column import DeepCrossFeatureColumnInfo, DeepFMFeatureColumnInfo
from model.deepcross import DeepAndCrossClassifier
from model.deepfm import DeepFMClassifier

#
# tf estimator wrapper class
# 
class CustomEstimator:
    def __init__(self, estimator, feature_column_info):
        self.estimator = estimator
        self.feature_column_info = feature_column_info
 
    def get_column_config(self):
        return self.feature_column_info.column_config

    def get_tf_estimator(self):
        return self.estimator

    def get_feature_column_info(self):
        return self.feature_column_info


def get_custom_estimator(model_type, model_config, column_config_file, optimizer,
   learning_rate, model_dir,  hvd=None, run_config=None):
   if model_type == "deepcross":
      feature_column_info = DeepCrossFeatureColumnInfo(column_config_file)

      estimator = DeepAndCrossClassifier(
         hidden_units=model_config.hidden_units,
         cross_layer_cnt=model_config.cross_layer_cnt,
         feature_columns=feature_column_info.feature_columns,
         position_bias_column=feature_column_info.position_bias_feature_colum,
         model_dir=model_dir,
         optimizer=optimizer,
         dropout=model_config.dropout_rate,
         batch_norm_enable=False,
         learning_rate=learning_rate,
         hvd=hvd,
         config=run_config)

   elif model_type == "deepfm":
      feature_column_info = DeepFMFeatureColumnInfo(column_config_file)

      estimator = DeepFMClassifier(
         hidden_units=model_config.hidden_units,
         position_bias_column=feature_column_info.position_bias_feature_column,
         numeric_columns=feature_column_info.numeric_feature_columns,
         categorical_columns=feature_column_info.categorical_columns,
         categorical_embedding_columns=\
             feature_column_info.categorical_embedding_feature_columns(model_config.embedding_size),
         embedding_size=model_config.embedding_size,
         model_dir=model_dir,
         optimizer=optimizer,
         dropout=model_config.dropout_rate,
         learning_rate=learning_rate,
         batch_norm_enable=False,
         hvd=hvd,
         config=run_config)
   else:
      raise ValueError("no suppported model")

   return  CustomEstimator(estimator, feature_column_info)
