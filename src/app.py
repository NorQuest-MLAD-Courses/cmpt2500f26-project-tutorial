"""
app.py — Flask REST API for the churn prediction service.

This module creates a small web server with Flask. A web server listens for
HTTP requests (for example from a browser, `curl` or another program) and
sends back responses. Each URL the server answers is called a *route* or
*endpoint*.

At this stage the API has a single endpoint:

    GET /health  ->  {"status": "ok"}

A health endpoint lets people and tools check that the service is running
before they send it real work.

How to run (from the project root):

    python src/app.py

then open http://127.0.0.1:5000/health in a browser, or run
`curl http://127.0.0.1:5000/health` in a second terminal.
"""

from flask import Flask, jsonify

# Create the Flask application object. `__name__` tells Flask which module
# the app lives in, so it can locate resources relative to this file. Every
# route below is registered on this `app` object.
app = Flask(__name__)


# The @app.route decorator connects a URL path and HTTP method to the Python
# function below it. When a GET request arrives at /health, Flask calls
# health() and sends its return value back to the client.
@app.route("/health", methods=["GET"])
def health():
    """Return a simple status check."""
    # jsonify turns a Python dict into a JSON response and sets the
    # Content-Type header to application/json. The status code defaults to 200.
    return jsonify({"status": "ok"})


# This block runs only when the file is executed directly
# (`python src/app.py`), not when it is imported, for example by the tests.
if __name__ == "__main__":
    import os
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
