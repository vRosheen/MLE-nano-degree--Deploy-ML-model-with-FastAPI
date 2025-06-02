from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    """
    Test root page
    """
    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == {
        "greeting": "Hello, this app predicts income (<=50K, >50K)."}


def test_post_predict_inf():
    """
    Test prediction when expected result is <=50K
    """

    r = client.post("/predict", json={
        "age": 38,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predict_sup():
    """
    Test prediction when expected result is >50K
    """
        
    r = client.post("/predict", json={
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
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
