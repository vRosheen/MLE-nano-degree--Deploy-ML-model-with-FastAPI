import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


features = {
    "age": 38,
    "workclass": "Federal-gov",
    "fnlgt": 292175,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Divorced",
    "occupation": "Exec-managerial",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

app_url = "https://fastapi-ml-model-34p9.onrender.com/predict"

response = requests.post(app_url, json=features)

logging.info("Sending POST request to Render app")
logging.info(f"Status code: {response.status_code}")
logging.info(f"Response text: {response.text}")
