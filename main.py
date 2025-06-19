from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig,DataTransformationConfig
from networksecurity.components.data_transformation import DataTransformation
import sys
if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig= DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiating Data Ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("data initiation completed")
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate Data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        print(dataingestionartifact)
        print(data_validation_artifact)
        logging.info("Data validation complete")
        logging.info(" Data Transformation starting")
        data_tranformation_config=DataTransformationConfig(trainingpipelineconfig)
        data_transformation=DataTransformation(data_validation_artifact,data_tranformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("Data Transformation complete")
        print(data_transformation_artifact)

    




    except Exception as e:
        raise CustomException(e,sys)
   