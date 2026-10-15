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

# .PHONY lists targets that are command names, not files. Without it, a
# file or folder with the same name as a target (for example "clean") in the
# project root would make Make think the target is already up to date, and
# it would skip the recipe. Every new target is added to this list.
.PHONY: venv preprocess train evaluate predict test mlflow-ui clean

# make venv
# Creates the project's virtual environment in .venv/ and installs every
# package listed in requirements.txt into it. Run it once after cloning the
# repository, and again whenever requirements.txt changes.
venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

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

# make clean
# Deletes generated files so you can start again from a clean state:
#   .venv              the virtual environment (recreate it with `make venv`)
#   models/*.pkl       saved models (recreate them with `make train`)
#   data/processed/*.csv  processed data (recreate it with `make preprocess`)
#   mlflow.db mlruns/  the local MLflow tracking database and artifact folder
clean:
	rm -rf .venv models/*.pkl data/processed/*.csv mlflow.db mlruns/
