import os
import joblib
import logging
import numpy as np
import pandas as pd
import uvicorn
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from src.ml.data import process_data
from src.ml.model import inference


logging.basicConfig(level=logging.INFO, format="%(message)s")


app = FastAPI()

def hyphenize(field: str):
    return field.replace("_", "-")


class ModelInput(BaseModel):
    model_config = ConfigDict(
        alias_generator=hyphenize,
        json_schema_extra={
            "example": {
                "age": 43,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 292175,
                "education": "Masters",
                "education-num": 14,
                "marital-status": "Divorced",
                "occupation": "Exec-managerial",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States",
            }
        }
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
async def say_hello():
    return {"greeting": "Hello, this app predicts income (<=50K, >50K)."}


@app.post("/predict")
async def predict(input: ModelInput):
    features = [
        "age", "workclass", "fnlgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country"
    ]

    cat_features = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race", "sex", "native-country"
    ]

    # Convert input to dataframe using field aliases
    input_dict = input.model_dump(by_alias=True)
    input_df = pd.DataFrame([[input_dict[feat] for feat in features]], columns=features)

    # Load model and transformers
    model = joblib.load("model/rf_model.pkl")
    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/lb.pkl")

    # Process and predict
    X, _, _, _ = process_data(input_df, categorical_features=cat_features,
                              training=False, encoder=encoder, lb=lb)
    prediction = lb.inverse_transform(inference(model, X))[0]

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)