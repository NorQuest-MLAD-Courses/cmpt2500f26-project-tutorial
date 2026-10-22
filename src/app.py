"""
app.py — Flask REST API for the churn prediction service.

This module creates a web server with Flask. A web server listens for HTTP
requests (for example from a browser, `curl` or another program) and sends
back responses. Each URL the server answers is called a *route* or
*endpoint*.

Endpoints:

    GET  /health      ->  {"status": "ok"}
    POST /v1/predict  ->  churn prediction for one customer, from model v1

When the module is loaded it reads two trained pipelines,
models/model_v1.pkl and models/model_v2.pkl, from disk. Both are complete
scikit-learn pipelines produced by the tuning step, so they accept cleaned
but unencoded customer data: the same column names and text values as the
raw CSV.

Input:  a JSON object with one customer's features (see REQUIRED_FEATURES).
Output: JSON with the predicted label ("Yes"/"No"), the probability of that
        label, and the model version that produced it.

How to run (from the project root):

    python src/app.py

then, in a second terminal, for example:

    curl http://127.0.0.1:5000/health
    curl -X POST http://127.0.0.1:5000/v1/predict \\
         -H "Content-Type: application/json" -d '{"tenure": 12, ...}'
"""

import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request

# Create the Flask application object. `__name__` tells Flask which module
# the app lives in, so it can locate resources relative to this file. Every
# route below is registered on this `app` object.
app = Flask(__name__)


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


# ---- Endpoints ----

# The @app.route decorator connects a URL path and HTTP method to the Python
# function below it. When a GET request arrives at /health, Flask calls
# health() and sends its return value back to the client.
@app.route("/health", methods=["GET"])
def health():
    """Return a simple status check."""
    # jsonify turns a Python dict into a JSON response and sets the
    # Content-Type header to application/json. The status code defaults to 200.
    return jsonify({"status": "ok"})


# POST is used because the client sends data (the customer record) in the
# body of the request.
@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    """Accept customer data as JSON, return a churn prediction from model v1."""
    # Parse the request body as JSON. With silent=True, an empty or malformed
    # body gives None instead of raising an error that Flask would turn into
    # an HTML error page, so we can reply with our own JSON error message.
    json_data = request.get_json(silent=True)
    if not json_data:
        # 400 Bad Request: the problem is in what the client sent.
        return jsonify({"error": "No input data provided"}), 400

    # Convert to DataFrame (single row)
    input_df = pd.DataFrame([json_data])

    # Select the columns in training order and run the model. If something
    # goes wrong (for example a missing column), return the error as JSON
    # with status 500 (Internal Server Error) instead of crashing.
    try:
        yhat = model_v1.predict(input_df[REQUIRED_FEATURES])
        # predict_proba returns one row per record and one column per class:
        # proba[0] is [P(class 0 = "No"), P(class 1 = "Yes")] for the first row.
        proba = model_v1.predict_proba(input_df[REQUIRED_FEATURES])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # yhat[0] is the predicted class (0 or 1) for our single row. Using it as
    # the column index, proba[0][yhat[0]] is the probability of the predicted
    # class. float() converts the NumPy number to a plain Python float, which
    # jsonify can serialise.
    return jsonify({
        "prediction": "Yes" if yhat[0] == 1 else "No",
        "probability": float(proba[0][yhat[0]]),
        "model_version": "v1",
    })


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
