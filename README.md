# Telco Customer Churn Prediction

Predicts customer churn using the Telco Customer Churn dataset.

## Setup

```bash
make venv
```

## Usage

```bash
make preprocess   # clean and encode raw data
make train        # train pipeline and save to models/
make evaluate     # evaluate saved pipeline on test split
make predict      # print predictions to screen
make test         # run test suite
```

## Project Structure

```
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_preprocess.py
│   └── test_train.py
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
├── models/
├── pytest.ini
├── Makefile
├── requirements.txt
└── README.md
```


## Data attribution

The dataset is the IBM Telco Customer Churn dataset, distributed on
Kaggle by user BlastChar
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
The original source is IBM Sample Data Sets
(https://github.com/IBM/telco-customer-churn-on-icp4d).

The proof-of-concept notebook used as the starting point for this
project was created by Bharti Prasad
(https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction).
