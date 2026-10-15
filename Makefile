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
.PHONY: setup venv preprocess train evaluate predict test mlflow-ui dvc-push dvc-pull clean

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
# The two `read -s -p` commands prompt for your DagsHub credentials. -p shows
# the prompt text and -s hides what you type, so the keys never appear on
# screen or in your shell history. "$$KEY_ID" is how a Makefile writes the
# shell variable $KEY_ID (Make itself would expand a single $).
# The commands are joined with `&& \` so they run in ONE shell; otherwise the
# variables set by `read` would be lost before `dvc remote modify` runs.
# `dvc remote modify origin --local ...` stores the keys in .dvc/config.local,
# which is git-ignored, so your secrets never end up in Git. The non-secret
# remote settings live in .dvc/config, which is committed.
# Finally `dvc pull` downloads the DVC-tracked files.
# A leading @ on a recipe line stops Make from printing the command itself;
# only its output is shown.
setup: venv
	@echo "Configuring DVC remote credentials ..."
	@read -s -p "DagsHub Access Key ID: " KEY_ID && echo "" && \
	 read -s -p "DagsHub Secret Access Key: " SECRET_KEY && echo "" && \
	 .venv/bin/dvc remote modify origin --local access_key_id "$$KEY_ID" && \
	 .venv/bin/dvc remote modify origin --local secret_access_key "$$SECRET_KEY"
	@echo "Pulling data from DVC remote ..."
	.venv/bin/dvc pull
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

# make clean
# Deletes generated files so you can start again from a clean state:
#   .venv              the virtual environment (recreate it with `make venv`)
#   models/*.pkl       saved models (recreate them with `make train`)
#   data/processed/*.csv  processed data (recreate it with `make preprocess`)
#   mlflow.db mlruns/  the local MLflow tracking database and artifact folder
clean:
	rm -rf .venv models/*.pkl data/processed/*.csv mlflow.db mlruns/
