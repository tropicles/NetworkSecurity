import os
import sys
import json
import numpy as np
from networksecurity.exception.exception import CustomException
import pymongo 
from networksecurity.logging import logger
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
import certifi
ca=certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    def cv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client= pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    try:
        FILE_PATH="Network_data\phisingData.csv"
        DATABASE="Mongdb"
        COLLECTION="NetworkData"
        networkobj=NetworkDataExtract()
        records=networkobj.cv_to_json_convertor(FILE_PATH)
        no_of_records= networkobj.insert_data_mongodb(records,DATABASE,COLLECTION)
        print(no_of_records)

    except Exception as e:
        raise CustomException(e,sys)
    


        