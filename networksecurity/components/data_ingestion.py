import os
import sys
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Connects to MongoDB, fetches all documents from the specified collection, 
        converts to a pandas DataFrame, cleans it, and returns it.
        """
        try:
            db_name = self.data_ingestion_config.database_name
            coll_name = self.data_ingestion_config.collection_name
            client = pymongo.MongoClient(MONGO_DB_URL)
            collection = client[db_name][coll_name]

            # Fetch documents and convert to DataFrame
            df = pd.DataFrame(list(collection.find()))

            # Drop Mongo's default _id column, if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)

            # Replace placeholder strings 'na' with actual NaN
            df.replace({'na': np.nan}, inplace=True)
            logging.info(f"Fetched {len(df)} records from MongoDB collection '{coll_name}' in database '{db_name}'")

            return df

        except Exception as e:
            raise CustomException(e, sys)
    
    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Writes the DataFrame to the feature store file path as CSV and returns it.
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)

            dataframe.to_csv(feature_store_path, index=False, header=True)
            logging.info(f"Exported data into feature store at: {feature_store_path}")

            return dataframe
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the DataFrame into train and test sets and writes them to disk.
        """
        try:
            ratio = self.data_ingestion_config.train_test_split_ratio
            train_set, test_set = train_test_split(dataframe, test_size=ratio)
            logging.info(f"Performed train-test split with test size = {ratio}")

            # Ensure output directories exist
            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_path), exist_ok=True)

            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)
            logging.info(f"Training data saved at: {train_path}")
            logging.info(f"Testing data saved at: {test_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process: fetch from DB, save feature store,
        split into train/test, and returns an artifact with file paths.
        """
        try:
            # Load data from MongoDB
            df = self.export_collection_as_dataframe()

            # Save raw features
            df = self.export_data_into_feature_store(df)

            # Split and save train/test
            self.split_data_as_train_test(df)

            # Build and return the ingestion artifact
            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info("Data ingestion completed successfully.")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)
