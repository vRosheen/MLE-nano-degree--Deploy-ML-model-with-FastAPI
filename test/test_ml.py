import os
from os import path
import logging
import joblib
import pytest
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.ml.data import process_data
from src.ml.model import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


@pytest.fixture()
def data():
    """
    Fixture - returns dataset
    """

    try:
        df = pd.read_csv("data/census_cleaned.csv")
        logging.info("Data fixture: SUCCESS")
        return df

    except FileNotFoundError as err:
        logging.error("Testing data fixture: The file wasn't found")
        raise err


@pytest.fixture()
def processed_data(data):
    """
    Fixture - returns processed dataset
    """

    cat_features = [
        "workclass",
        "education",  # may be remove, duplicate of education-num
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    try:
        X, y, encoder, lb = process_data(
            data, categorical_features=cat_features, label="salary", training=True
        )
        logging.info("Processed process_data fixture: SUCCESS")
        return X, y, encoder, lb

    except Exception as e:
        logging.error("Testing process_data fixture: FAILED")
        raise e
    


def test_data_process(processed_data):
    """ 
    Test data processing 
    """

    X, y, encoder, lb = processed_data
    try:

        assert isinstance(encoder, OneHotEncoder)
        assert isinstance(lb, LabelBinarizer)
        assert len(X) == len(y)

        joblib.load('model/lb.pkl')
        joblib.load('model/encoder.pkl')
        logging.info("Testing process_data: SUCCESS")

    except Exception as e:
        logging.info("Testing process_data: FAILED")
        raise e


def test_train_model(processed_data):
    """ 
    Test model training 
    """

    X, y, _, _ = processed_data
    try:
        model = train_model(X, y)
        assert isinstance(model, RandomForestClassifier)

        joblib.load('model/rf_model.pkl')
        logging.info("Testing train_model: SUCCESS")

    except Exception as e:
        logging.info("Testing train_model: FAILED")
        raise e
