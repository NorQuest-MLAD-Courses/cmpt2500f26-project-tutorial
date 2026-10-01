.PHONY: venv preprocess train evaluate predict test clean

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

preprocess:
	.venv/bin/python src/preprocess.py

train:
	.venv/bin/python src/train.py

evaluate:
	.venv/bin/python src/evaluate.py

predict:
	.venv/bin/python src/predict.py

test:
	.venv/bin/pytest

clean:
	rm -rf .venv models/*.pkl data/processed/*.csv
