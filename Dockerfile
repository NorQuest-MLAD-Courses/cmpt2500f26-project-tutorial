# Dockerfile for the churn-prediction API image.
# A Dockerfile is a recipe for building an image: each instruction below
# adds one layer on top of the previous one. Build it with
# `make docker-build` (docker build -t churn-api .).
# Reference: https://docs.docker.com/reference/dockerfile/

# ---- Base image ----
# Start from the official Python 3.11 image. The "slim" variant is a
# minimal Debian system with Python installed, which keeps the image small.
FROM python:3.11-slim

# ---- Set working directory ----
# Every following instruction (and the running container) works inside
# /app, as if we had run `cd /app`. Docker creates the folder if needed.
WORKDIR /app

# ---- Install Python dependencies ----
# Copy only requirements first so Docker caches this layer.
# Re-running `docker build` after a code change skips the install
# unless requirements.txt itself changed.
# Why this order: Docker reuses a cached layer as long as that layer and
# every layer before it are unchanged. Code in src/ changes often and
# requirements.txt rarely, so installing packages BEFORE copying the code
# keeps the slow pip install cached across most rebuilds.
# --no-cache-dir stops pip from keeping its download cache in the image.
# There is no virtual environment here: the container itself is isolated
# and runs only this one project.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code and models ----
# Copy the application code and the trained models (models/model_v1.pkl and
# models/model_v2.pkl must exist before you build) into /app.
COPY src/ src/
COPY models/ models/

# ---- Expose the API port ----
# Documents that the container listens on port 5000. It does not publish
# the port; that happens at run time with `docker run -p 5000:5000`.
EXPOSE 5000

# ---- Run with Gunicorn (production server) ----
# The command the container runs when it starts.
#   gunicorn            production web server (Flask's own server is for development)
#   --bind 0.0.0.0:5000 listen on all network interfaces, so connections from
#                       outside the container reach it, on port 5000
#   --chdir src         change into src/ first, so `app:app` finds src/app.py
#                       and the relative model paths in app.py resolve correctly
#   app:app             the module app.py and the Flask object named app inside it
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "src", "app:app"]
