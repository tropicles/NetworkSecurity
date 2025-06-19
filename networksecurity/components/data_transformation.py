import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        logging.info("Creating data transformer pipeline with KNN Imputer")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Initiating data transformation")
        try:
            # Read validated train and test data
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path)

            # Separate input features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Create and fit transformer on training data
            processor = self.get_data_transformer_object()
            transformer = processor.fit(input_feature_train_df)

            # Transform both train and test features
            transformed_input_train_feature = transformer.transform(input_feature_train_df)
            transformed_input_test_feature = transformer.transform(input_feature_test_df)

            # Combine transformed features with their respective targets
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save numpy arrays and transformer object
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                transformer
            )

            # Prepare artifact to return (positional args to match constructor signature)
            data_transformation_artifact = DataTransformationArtifact(
                self.data_transformation_config.transformed_train_file_path,
                self.data_transformation_config.transformed_test_file_path,
                self.data_transformation_config.transformed_object_file_path
            )

            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
