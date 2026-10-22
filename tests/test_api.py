"""
test_api.py — Automated tests for the Flask API endpoints.

These tests call the API in src/app.py without starting a real server.
Flask's *test client* sends fake HTTP requests straight to the app inside
the same Python process and returns the responses, so the tests are fast and
need no free network port.

Run from the project root with:

    pytest

Each test sends one request and checks the status code and the JSON body.

Importing src.app loads the trained models from models/model_v1.pkl and
models/model_v2.pkl, so those files must be present (for example after
`dvc pull`) before the tests run. The prediction tests use real models but
a made-up customer (VALID_PAYLOAD below).
"""

import pytest
# `src.app` can be imported because pytest.ini sets `pythonpath = .`, which
# puts the project root on the import path. The Flask object is renamed to
# flask_app so that the name `app` stays free.
from src.app import app as flask_app


# A pytest *fixture* is a function that prepares something a test needs.
# Any test that lists `client` as a parameter gets the value this fixture
# yields; pytest runs the fixture again for every such test, so each test
# starts with a fresh client.
@pytest.fixture
def client():
    """Create a Flask test client."""
    # TESTING mode makes Flask pass errors through to the test instead of
    # turning them into a generic 500 page, so failures are easier to read.
    flask_app.config["TESTING"] = True
    # The `with` block keeps the client open while the test runs. `yield`
    # hands the client to the test; the code after it (closing the `with`
    # block) runs when the test has finished.
    with flask_app.test_client() as client:
        yield client


# A valid customer payload for reuse across tests
# One customer with every feature the API requires, using the same column
# names and raw text values as the original CSV. Tests send it as the JSON
# body of a request, or change a copy of it to make invalid input. The values
# are made up; the tests only check the shape of the response, not whether
# the model predicts churn for this customer.
VALID_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 59.95,
    "TotalCharges": 720.50,
    "Contract": "One year",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "InternetService": "DSL",
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "PaperlessBilling": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
}


# ---- Step 9: Health ----

def test_health(client):
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    # response.json parses the JSON body into a Python dict.
    assert response.json["status"] == "ok"


# ---- Step 10: v1 single prediction ----

def test_v1_single_prediction(client):
    """POST /v1/predict with valid data returns a prediction."""
    # json=... serialises the dict to JSON and sets the
    # Content-Type: application/json header, as a real client would.
    response = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "probability" in response.json
    assert response.json["model_version"] == "v1"
    # The label must be one of the two strings the API promises.
    assert response.json["prediction"] in ("Yes", "No")


# ---- Step 11: Validation, batch, v2 ----

def test_v1_batch_prediction(client):
    """POST /v1/predict with a list returns a list of predictions."""
    # Sending a JSON array (a Python list) asks for a batch prediction.
    # The reply must also be a list, with one result per record.
    response = client.post("/v1/predict", json=[VALID_PAYLOAD, VALID_PAYLOAD])
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert len(response.json) == 2
    assert response.json[0]["model_version"] == "v1"


def test_v1_missing_features(client):
    """POST /v1/predict with missing fields returns 400."""
    # Only one of the required features is present, so validation must
    # reject the request with 400 Bad Request before the model is called.
    response = client.post("/v1/predict", json={"tenure": 10})
    assert response.status_code == 400
    assert "Missing required features" in response.json["error"]


def test_v1_wrong_type(client):
    """POST /v1/predict with wrong data type returns 400."""
    # Change a copy so VALID_PAYLOAD stays valid for the other tests.
    # A number sent as text must be rejected by the type check.
    bad = VALID_PAYLOAD.copy()
    bad["tenure"] = "twelve"
    response = client.post("/v1/predict", json=bad)
    assert response.status_code == 400
    assert "Invalid type for tenure" in response.json["error"]


def test_v1_empty_body(client):
    """POST /v1/predict with no body returns 400."""
    # An empty body with a JSON content type: request.get_json(silent=True)
    # returns None, and the endpoint must answer 400. data=... sends the raw
    # body as given, unlike json=..., which would serialise a Python value.
    response = client.post("/v1/predict",
                           data="",
                           content_type="application/json")
    assert response.status_code == 400


def test_v2_single_prediction(client):
    """POST /v2/predict returns a prediction with model_version v2."""
    # The v2 endpoint uses the second model; the response says which one.
    response = client.post("/v2/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json["model_version"] == "v2"
    assert response.json["prediction"] in ("Yes", "No")


# ---- Step 12: Info / documentation ----

def test_info(client):
    """GET /info returns API description with endpoints and input format."""
    response = client.get("/info")
    assert response.status_code == 200
    # Check the structure of the reply: the endpoint list and the input
    # format section with both feature lists and an example record.
    assert "endpoints" in response.json
    assert "required_input_format" in response.json
    assert "numerical_features" in response.json["required_input_format"]
    assert "categorical_features" in response.json["required_input_format"]
    assert "example" in response.json["required_input_format"]
