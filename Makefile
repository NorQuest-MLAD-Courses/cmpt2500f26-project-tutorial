# Makefile for the churn-prediction project.
#
# A Makefile gives short names ("targets") to the commands we run often.
# Type `make <target>` in the project root, for example `make venv`.
# Each target is a name followed by a colon; the indented lines under it
# (the "recipe") are shell commands that Make runs one after another.
# Recipe lines MUST start with a real tab character, not spaces.
# Manual: https://www.gnu.org/software/make/manual/make.html
#
# Targets call programs inside the virtual environment by their full path
# (for example .venv/bin/pip) instead of activating the environment.
# Each recipe line runs in its own new shell, so `source .venv/bin/activate`
# on one line would be forgotten by the next. The full path always uses the
# project's environment, never the system Python.

# Make runs recipes with /bin/sh by default. On many Linux systems that is
# a minimal shell (dash) whose `read` has no -s option. The setup
# target below needs bash's `read -s -p`, so we tell Make to use bash.
SHELL := /bin/bash

# .PHONY lists targets that are command names, not files. Without it, a
# file or folder with the same name as a target (for example "clean") in the
# project root would make Make think the target is already up to date, and
# it would skip the recipe. Every new target is added to this list.
.PHONY: setup venv preprocess train tune evaluate predict test api mlflow-ui dvc-push dvc-pull docker-build docker-run docker-stop compose-up compose-down compose-train compose-tune docker-tag docker-push deploy clean

# make venv
# Creates the project's virtual environment in .venv/ and installs every
# package listed in requirements.txt into it. Run it once after cloning the
# repository, and again whenever requirements.txt changes.
venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# make setup
# One-command setup for a fresh clone or a new Codespace.
# "setup: venv" means setup depends on venv: Make runs the venv target first,
# then the commands below.
# A leading @ on a recipe line stops Make from printing the command itself;
# only its output is shown.
#
# 1. Endpoint check. DagsHub's repository page shows two different S3
#    endpoints. DVC needs https://dagshub.com/<user>/<repo>.s3; the one in the
#    "S3 Connection Credentials" panel (.../api/v1/repo-buckets/s3/<user>) is
#    for the "Use as S3" bucket and makes `dvc pull` fail with a 404. We stop
#    early with a clear message if .dvc/config has that one. `exit 1` makes
#    the target fail, so Make does not run the steps after it.
# 2. Token. DagsHub uses the same token as both the access key ID and the
#    secret access key, so we ask for it once and store it in both places.
#    `read -s -p` shows the prompt (-p) and hides what you type or paste
#    (-s), so the token never appears on screen or in your shell history.
#    "$$TOKEN" is how a Makefile writes the shell variable $TOKEN (Make itself
#    would expand a single $). $$STORED holds the token that is already in
#    .dvc/config.local, or nothing if there is none. If one is stored (you
#    ran setup before in this Codespace), pressing Enter keeps it; pasting a
#    new token replaces it.
#    The lines are joined with `\` so they run in ONE shell; otherwise the
#    variable set by `read` would be lost before `dvc remote modify` runs.
#    `dvc remote modify origin --local ...` stores the token in
#    .dvc/config.local, which is git-ignored, so it never ends up in Git. The
#    non-secret remote settings live in .dvc/config, which is committed.
# 3. Finally `dvc pull` downloads the DVC-tracked files.
# 4. The gcloud block installs the Google Cloud command-line tool if it is
#    not already on your PATH (`command -v gcloud` checks for it). It is
#    needed for `make deploy`. After a fresh install you must restart the
#    shell and then log in and choose your Google Cloud project, as the
#    messages say.
setup: venv
	@if .venv/bin/dvc config remote.origin.endpointurl \
			| grep -q "repo-buckets"; then \
		echo "ERROR: wrong DagsHub endpoint in .dvc/config."; \
		echo "It must look like https://dagshub.com/<user>/<repo>.s3"; \
		exit 1; \
	fi
	@echo "Configuring DVC remote credentials ..."
	@STORED=$$(.venv/bin/dvc config --local \
			remote.origin.access_key_id 2> /dev/null); \
	if [ -n "$$STORED" ]; then \
		echo "A token is already stored. Press Enter to keep it."; \
	fi; \
	read -s -p "DagsHub token: " TOKEN && echo "" && \
	if [ -n "$$TOKEN" ]; then \
		.venv/bin/dvc remote modify origin --local \
			access_key_id "$$TOKEN" && \
		.venv/bin/dvc remote modify origin --local \
			secret_access_key "$$TOKEN"; \
	elif [ -z "$$STORED" ]; then \
		echo "No token entered. Run 'make setup' again."; \
		exit 1; \
	fi
	@echo "Pulling data from DVC remote ..."
	.venv/bin/dvc pull
	@echo ""
	@echo "Installing Google Cloud CLI ..."
	@if ! command -v gcloud &> /dev/null; then \
		echo "gcloud not found — installing ..."; \
		curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts; \
		echo ""; \
		echo "gcloud installed. Restart your shell (exec -l $$SHELL) then run:"; \
		echo "  gcloud auth login"; \
		echo "  gcloud config set project <YOUR_PROJECT_ID>"; \
	else \
		echo "gcloud already installed: $$(gcloud --version 2>&1 | head -1)"; \
	fi
	@echo ""
	@echo "Setup complete. Run 'make test' to verify."

# make preprocess
# Cleans the raw CSV and writes the processed data to data/processed/.
preprocess:
	.venv/bin/python src/preprocess.py

# make train
# Trains the model on the processed data and saves it in models/.
train:
	.venv/bin/python src/train.py

# make tune
# Runs the hyperparameter search in src/tune.py, logs every candidate to
# MLflow and saves the two best pipelines as models/model_v1.pkl and
# models/model_v2.pkl (the two API versions).
tune:
	.venv/bin/python src/tune.py

# make evaluate
# Loads the saved model and prints its scores on the test split.
evaluate:
	.venv/bin/python src/evaluate.py

# make predict
# Loads the saved model and makes predictions for new customers.
predict:
	.venv/bin/python src/predict.py

# make test
# Runs the automated tests in tests/ with pytest (settings in pytest.ini).
# Run it after every change; all tests should pass before you commit.
test:
	.venv/bin/pytest

# make api
# Starts the Flask API on your machine (outside Docker) for quick testing.
# Stop it with Ctrl+C.
api:
	.venv/bin/python src/app.py

# make mlflow-ui
# Opens the MLflow web interface so you can browse and compare runs.
# --backend-store-uri points it at the same SQLite file the training code
# writes to: sqlite:///mlflow.db (three slashes = a path relative to the
# current folder). Without it the UI would look in the wrong place and show
# no runs. Stop it with Ctrl+C.
mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

# make dvc-push
# Uploads DVC-tracked files to the DagsHub remote.
# Run it after `dvc add` and before `git push`, so that teammates who pull
# the new .dvc files can also download the matching data.
dvc-push:
	.venv/bin/dvc push

# make dvc-pull
# Downloads the DVC-tracked files that match the current .dvc files in Git.
dvc-pull:
	.venv/bin/dvc pull

# make docker-build
# Builds a Docker image from the Dockerfile in this folder (the final "."
# is the build context) and names it churn-api with -t.
docker-build:
	docker build -t churn-api .

# make docker-run
# Starts a container from the churn-api image.
#   -d               run in the background (detached)
#   --name churn-api give the container a fixed name so docker-stop can find it
#   -p 5000:5000     map port 5000 on your machine to port 5000 in the container
docker-run:
	docker run -d --name churn-api -p 5000:5000 churn-api

# make docker-stop
# Stops the churn-api container and then removes it, so the name is free
# for the next `make docker-run`.
docker-stop:
	docker stop churn-api && docker rm churn-api

# make compose-up
# Starts every service in docker-compose.yml. --build rebuilds the images
# first if their files changed; -d runs the stack in the background.
compose-up:
	docker compose up --build -d

# make compose-down
# Stops and removes the Compose containers and network. Named volumes
# (such as the MLflow data) are kept.
compose-down:
	docker compose down

# make compose-train
# Runs the training script inside a one-off container built from the api
# service, so it talks to the MLflow container over the Compose network.
# --rm deletes that one-off container when the script finishes.
compose-train:
	docker compose run --rm api python src/train.py

# make compose-tune
# Runs the tuning script inside a one-off api container (removed afterwards
# by --rm), logging to the MLflow container.
compose-tune:
	docker compose run --rm api python src/tune.py

# make docker-tag DOCKER_USER=<your-docker-hub-username>
# (or: DOCKER_USER=<your-username> make docker-tag)
# Gives the local images a second name in the form <user>/<image>:latest,
# which is the name Docker Hub expects. $(DOCKER_USER) is a Make variable
# that you set on the command line or in the environment.
# `$(shell docker compose images mlflow -q)` runs a command when Make expands
# the recipe and inserts its output: the ID of the image Compose built for
# the mlflow service (run `make compose-up` first so that image exists).
docker-tag:
	docker tag churn-api $(DOCKER_USER)/churn-api:latest
	docker tag $(shell docker compose images mlflow -q) $(DOCKER_USER)/churn-mlflow:latest

# make docker-push DOCKER_USER=<your-docker-hub-username>
# Uploads both tagged images to Docker Hub. Run `docker login` once first.
docker-push:
	docker push $(DOCKER_USER)/churn-api:latest
	docker push $(DOCKER_USER)/churn-mlflow:latest

# make deploy DOCKER_USER=<your-docker-hub-username>
# Deploys the API image from Docker Hub to Google Cloud Run.
#   --image docker.io/...    Cloud Run needs the full registry address; a short
#                            name such as user/churn-api is not enough, so we
#                            add the docker.io/ prefix for Docker Hub
#   --platform managed       fully managed Cloud Run (Google runs the servers)
#   --region us-central1     the data centre region to deploy to
#   --port 5000              the port the container listens on (Gunicorn)
#   --allow-unauthenticated  anyone with the URL can call the API
# The backslashes continue one long shell command over several lines.
deploy:
	gcloud run deploy churn-api \
		--image docker.io/$(DOCKER_USER)/churn-api:latest \
		--platform managed \
		--region us-central1 \
		--port 5000 \
		--allow-unauthenticated

# make clean
# Deletes generated files so you can start again from a clean state:
#   .venv              the virtual environment (recreate it with `make venv`)
#   models/*.pkl       saved models (recreate them with `make train`)
#   data/processed/*.csv  processed data (recreate it with `make preprocess`)
#   mlflow.db mlruns/  the local MLflow tracking database and artifact folder
#   logs/              API log files
clean:
	rm -rf .venv models/*.pkl data/processed/*.csv mlflow.db mlruns/ logs/
