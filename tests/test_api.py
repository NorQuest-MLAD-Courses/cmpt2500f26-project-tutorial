"""
test_api.py — Automated tests for the Flask API endpoints.

These tests call the API in src/app.py without starting a real server.
Flask's *test client* sends fake HTTP requests straight to the app inside
the same Python process and returns the responses, so the tests are fast and
need no free network port.

Run from the project root with:

    pytest

Each test sends one request and checks the status code and the JSON body.
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


# ---- Step 9: Health ----

def test_health(client):
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    # response.json parses the JSON body into a Python dict.
    assert response.json["status"] == "ok"
