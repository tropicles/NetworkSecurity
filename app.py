import sys, os
import certifi
from dotenv import load_dotenv
import pymongo
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.responses import Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
ca = certifi.where()

# Initialize MongoDB client
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

# Initialize FastAPI app
app = FastAPI()

# CORS settings
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response(content="Training is successful", status_code=status.HTTP_200_OK)
    except Exception as e:
        raise CustomException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Read uploaded CSV into DataFrame
        df = pd.read_csv(file.file)

        # Load preprocessing pipeline and model
        preprocessor = load_object("final_model/preprocessing.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Make predictions
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        # Save output and render HTML table
        output_path = os.path.join('prediction_output', 'output.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        table_html = df.to_html(classes='table table-striped', index=False)

        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
