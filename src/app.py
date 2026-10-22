"""
app.py — Flask REST API for the churn prediction service.

This module creates a web server with Flask. A web server listens for HTTP
requests (for example from a browser, `curl` or another program) and sends
back responses. Each URL the server answers is called a *route* or
*endpoint*.

Endpoints:

    GET  /health      ->  {"status": "ok"}
    POST /v1/predict  ->  churn prediction(s) from model v1
    POST /v2/predict  ->  churn prediction(s) from model v2
    GET  /info        ->  description of the API and its input format
    GET  /apidocs/    ->  interactive Swagger documentation (from Flasgger)

When the module is loaded it reads two trained pipelines,
models/model_v1.pkl and models/model_v2.pkl, from disk. Both are complete
scikit-learn pipelines produced by the tuning step, so they accept cleaned
but unencoded customer data: the same column names and text values as the
raw CSV.

Input:  a JSON object with one customer's features (see REQUIRED_FEATURES),
        or a JSON array of such objects for a batch prediction.
Output: for each customer, JSON with the predicted label ("Yes"/"No"), the
        probability of that label, and the model version that produced it.
        A batch request gets a JSON array back. Invalid input gets a JSON
        error message with status 400.

How to run (from the project root):

    python src/app.py

then, in a second terminal, for example:

    curl http://127.0.0.1:5000/health
    curl http://127.0.0.1:5000/info
    curl -X POST http://127.0.0.1:5000/v1/predict \\
         -H "Content-Type: application/json" -d '{"tenure": 12, ...}'
"""

import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger

# Create the Flask application object. `__name__` tells Flask which module
# the app lives in, so it can locate resources relative to this file. Every
# route below is registered on this `app` object.
app = Flask(__name__)
# Flasgger adds interactive API documentation (Swagger UI) to the app.
# It reads the part of each route's docstring after the `---` line, which is
# written in YAML using the Swagger/OpenAPI format, and builds a web page at
# /apidocs/ that lists every endpoint, its expected input and its possible
# responses. Developers can also send test requests from that page. The raw
# specification is served as JSON at /apispec_1.json.
swagger = Swagger(app)


# ---- Load models at startup ----
# The models are loaded once, when this module is first imported, and kept
# in memory for the life of the process. Loading a model from disk on every
# request would make each prediction much slower.

def load_model(path):
    """Load a pickled pipeline from disk."""
    # "rb" = read binary; pickle files are bytes, not text.
    with open(path, "rb") as f:
        return pickle.load(f)


# Build the models/ path relative to this file (src/app.py -> ../models), so
# the server finds the models no matter which folder it is started from.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model_v1 = load_model(os.path.join(MODEL_DIR, "model_v1.pkl"))
model_v2 = load_model(os.path.join(MODEL_DIR, "model_v2.pkl"))

print(f"Loaded model_v1 and model_v2 from {MODEL_DIR}")


# ---- Feature definitions ----
# The columns the pipelines were trained on. Each pipeline scales the
# numeric columns and turns the categorical (text) columns into numbers
# itself, so the client sends raw values such as "One year".

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Every request must provide all of these columns. The list order is also
# the column order we pass to the model.
REQUIRED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ---- Validation ----
# Checking the input before it reaches the model means a client with a
# mistake (a missing field, a number sent as text) gets a clear message that
# names the problem, instead of a vague error from deep inside scikit-learn.

def validate_input(record):
    """
    Check that a single record has all required features with correct types.
    Returns (error_message, status_code) or (None, 200) if valid.
    """
    # Collect every missing feature, not just the first, so the client can
    # fix them all in one go.
    missing = [f for f in REQUIRED_FEATURES if f not in record]
    if missing:
        # 400 Bad Request: the client sent incomplete data.
        return f"Missing required features: {', '.join(missing)}", 400

    # Numeric features must be JSON numbers (Python int or float), not text
    # such as "12".
    for feat in NUMERICAL_FEATURES:
        if not isinstance(record[feat], (int, float)):
            # type(...).__name__ gives a readable name such as "str".
            return (f"Invalid type for {feat}: expected a number, "
                    f"got {type(record[feat]).__name__}"), 400

    # Categorical features must be strings such as "Yes" or "One year".
    for feat in CATEGORICAL_FEATURES:
        if not isinstance(record[feat], str):
            return (f"Invalid type for {feat}: expected a string, "
                    f"got {type(record[feat]).__name__}"), 400

    # No problems found. 200 means OK; the caller only checks `error`.
    return None, 200


# ---- Prediction helper ----
# Both /v1/predict and /v2/predict need the same steps (validate, predict,
# format the answer). Keeping them in one function means the two endpoints
# differ only in which model they pass in.

def run_prediction(model, model_label, json_data):
    """
    Validate, predict, and format results.
    Handles both single records and batches.
    """
    # Batch support: the client may send one JSON object (a dict) or a JSON
    # array of objects (a list). We turn both into a list of records so the
    # rest of the function has a single code path, and remember which form
    # came in so the reply has the same shape.
    is_batch = isinstance(json_data, list)
    records = json_data if is_batch else [json_data]

    # Validate every record before touching the model
    # If any record is invalid, the whole request is rejected and no
    # predictions are returned.
    for record in records:
        error, status = validate_input(record)
        if error:
            return jsonify({"error": error}), status

    # Build one DataFrame row per record, keep only the model's columns in
    # training order, and predict all rows in one call (faster than looping).
    # Any unexpected failure becomes a JSON error with status 500
    # (Internal Server Error), since the input already passed validation.
    try:
        input_df = pd.DataFrame(records)[REQUIRED_FEATURES]
        yhat = model.predict(input_df)
        # predict_proba returns one row per record and one column per class:
        # proba[i] is [P(class 0 = "No"), P(class 1 = "Yes")] for record i.
        proba = model.predict_proba(input_df)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Build one result per record. yhat[i] is the predicted class (0 or 1);
    # using it as the column index, proba[i][yhat[i]] is the probability of
    # the predicted class. float() converts the NumPy number to a plain
    # Python float, which jsonify can serialise.
    results = []
    for i in range(len(yhat)):
        results.append({
            "prediction": "Yes" if yhat[i] == 1 else "No",
            "probability": float(proba[i][yhat[i]]),
            "model_version": model_label,
        })

    # A batch request gets a list back; a single record gets a single object.
    return jsonify(results if is_batch else results[0])


# ---- Endpoints ----

# The @app.route decorator connects a URL path and HTTP method to the Python
# function below it. When a GET request arrives at /health, Flask calls
# health() and sends its return value back to the client.
@app.route("/health", methods=["GET"])
def health():
    """
    Health Check
    ---
    responses:
      200:
        description: API is alive and running.
    """
    # jsonify turns a Python dict into a JSON response and sets the
    # Content-Type header to application/json. The status code defaults to 200.
    return jsonify({"status": "ok"})


# /info is documentation for people reading the API with a browser or curl:
# it lists the endpoints and the input format a prediction request needs.
# The feature lists come from the constants above, so they always match what
# validate_input checks.
@app.route("/info", methods=["GET"])
def info():
    """
    API Information
    Returns a description of the API, available endpoints,
    and the expected input format for predictions.
    ---
    responses:
      200:
        description: API information and usage guide.
    """
    return jsonify({
        "message": "Telco Customer Churn Prediction API",
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "predict_v1": "POST /v1/predict",
            "predict_v2": "POST /v2/predict",
            "docs": "GET /apidocs/",
        },
        "required_input_format": {
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "example": {
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
            },
        },
    })


# POST is used because the client sends data (the customer record) in the
# body of the request.
@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    """
    Predict churn using model v1
    ---
    tags:
      - Predictions
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: >
          Customer data. Send a single JSON object or a list of objects
          for batch prediction.
        schema:
          type: object
          properties:
            tenure:
              type: integer
              example: 12
            MonthlyCharges:
              type: number
              example: 59.95
            TotalCharges:
              type: number
              example: 720.50
            Contract:
              type: string
              example: "One year"
            gender:
              type: string
              example: "Female"
            SeniorCitizen:
              type: string
              example: "No"
            Partner:
              type: string
              example: "Yes"
            Dependents:
              type: string
              example: "No"
            PhoneService:
              type: string
              example: "Yes"
            MultipleLines:
              type: string
              example: "No"
            InternetService:
              type: string
              example: "DSL"
            OnlineSecurity:
              type: string
              example: "No"
            OnlineBackup:
              type: string
              example: "Yes"
            DeviceProtection:
              type: string
              example: "No"
            TechSupport:
              type: string
              example: "No"
            StreamingTV:
              type: string
              example: "No"
            StreamingMovies:
              type: string
              example: "No"
            PaperlessBilling:
              type: string
              example: "Yes"
            PaymentMethod:
              type: string
              example: "Electronic check"
    responses:
      200:
        description: Prediction successful.
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: "No"
            probability:
              type: number
              example: 0.9431
            model_version:
              type: string
              example: "v1"
      400:
        description: Invalid input data.
      500:
        description: Internal server error.
    """
    # Parse the request body as JSON. With silent=True, an empty or malformed
    # body gives None instead of raising an error that Flask would turn into
    # an HTML error page, so we can reply with our own JSON error message.
    json_data = request.get_json(silent=True)
    if not json_data:
        # 400 Bad Request: the problem is in what the client sent.
        return jsonify({"error": "No input data provided"}), 400
    return run_prediction(model_v1, "v1", json_data)


# API versioning: /v2/predict serves the second model. Existing clients keep
# calling /v1/predict and get the same behaviour, while new clients can move
# to v2 when they are ready. The code matches predict_v1 except for the model.
@app.route("/v2/predict", methods=["POST"])
def predict_v2():
    """
    Predict churn using model v2
    ---
    tags:
      - Predictions
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: >
          Customer data. Send a single JSON object or a list of objects
          for batch prediction.
        schema:
          type: object
          properties:
            tenure:
              type: integer
              example: 12
            MonthlyCharges:
              type: number
              example: 59.95
            TotalCharges:
              type: number
              example: 720.50
            Contract:
              type: string
              example: "One year"
            gender:
              type: string
              example: "Female"
            SeniorCitizen:
              type: string
              example: "No"
            Partner:
              type: string
              example: "Yes"
            Dependents:
              type: string
              example: "No"
            PhoneService:
              type: string
              example: "Yes"
            MultipleLines:
              type: string
              example: "No"
            InternetService:
              type: string
              example: "DSL"
            OnlineSecurity:
              type: string
              example: "No"
            OnlineBackup:
              type: string
              example: "Yes"
            DeviceProtection:
              type: string
              example: "No"
            TechSupport:
              type: string
              example: "No"
            StreamingTV:
              type: string
              example: "No"
            StreamingMovies:
              type: string
              example: "No"
            PaperlessBilling:
              type: string
              example: "Yes"
            PaymentMethod:
              type: string
              example: "Electronic check"
    responses:
      200:
        description: Prediction successful.
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: "No"
            probability:
              type: number
              example: 0.9431
            model_version:
              type: string
              example: "v2"
      400:
        description: Invalid input data.
      500:
        description: Internal server error.
    """
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "No input data provided"}), 400
    return run_prediction(model_v2, "v2", json_data)


# This block runs only when the file is executed directly
# (`python src/app.py`), not when it is imported, for example by the tests.
if __name__ == "__main__":
    # Read the port from the PORT environment variable so it can be changed
    # without editing code (useful if 5000 is already in use); default 5000.
    port = int(os.environ.get("PORT", 5000))
    # Start Flask's built-in development server.
    #   debug=True       reloads the server when the file changes and shows
    #                    detailed error pages; for development only.
    #   host="0.0.0.0"   listens on all network interfaces, not just
    #                    localhost, so the server is reachable from outside
    #                    the machine or container (needed in Codespaces).
    app.run(debug=True, host="0.0.0.0", port=port)
