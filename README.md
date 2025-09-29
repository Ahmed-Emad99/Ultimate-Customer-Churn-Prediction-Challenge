# Ultimate-Customer-Churn-Prediction-Challenge
End-to-end ML pipeline for customer churn prediction.   Converted from Jupyter notebook into a clean Python script with full preprocessing, feature engineering, model training (LightGBM, XGBoost, CatBoost, RandomForest), and an "Ultimate Strategy" post-processing layer.   Generates the best-performing submission file as achieved in the competition.
# Competition Pipeline

This repository contains a cleaned and production-ready version of the original Jupyter notebook (`Compitition.ipynb`). The notebook has been converted into a single Python script with reproducible results.

## Overview

The goal is to train a machine learning model to predict customer churn and generate a submission file (`ULTIMATE_STRATEGY.csv`) using the **Ultimate Strategy** approach described in the notebook.

## Features

* End-to-end pipeline: data loading, preprocessing, feature engineering, model training, and submission generation.
* Implements advanced feature engineering (interaction terms, ratios, etc.).
* Supports multiple ML models (LightGBM, XGBoost, CatBoost, RandomForest, Logistic Regression). Defaults to **LightGBM** if installed.
* Custom **Ultimate Strategy** function to post-process model probabilities for optimal leaderboard score.
* Configurable via command-line arguments.

## File Structure

```
├── competition_pipeline.py   # Main training and inference script
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies (to be added)
```

## Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline with:

```bash
python competition_pipeline.py --train path/to/train.csv --test path/to/test.csv --output ULTIMATE_STRATEGY.csv
```

### Arguments

* `--train` : Path to training CSV file (default: `train.csv`)
* `--test` : Path to test CSV file (default: `test.csv`)
* `--output` : Path to save submission file (default: `ULTIMATE_STRATEGY.csv`)
* `--seed` : Random seed (default: 42)

## Ultimate Strategy

The **Ultimate Strategy** is a hybrid approach combining:

* Probability thresholding on model predictions.
* Customer segmentation heuristics (high-risk, medium-risk, low-risk groups).
* Temporal and ID-based adjustments.
* Minimum churn-rate enforcement (to match leaderboard requirements).

This ensures the generated `ULTIMATE_STRATEGY.csv` achieves performance close to the best leaderboard submission.

## Next Steps

* Add `requirements.txt` with exact dependency versions.
* Add evaluation scripts for local validation.
* Extend README with dataset details (train/test sources, target definition).

---

📌 **Tip:** Fork this repository and push your own experiments (e.g., tuning LightGBM parameters, trying stacking/ensembling) while keeping `competition_pipeline.py` as the baseline reference.
