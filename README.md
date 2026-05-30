# Traffic Demand Prediction (Flipkart Grid)

Predict traffic `demand` from geohash, time, road, and weather features. Evaluation metric: `max(0, 100 × R²)`.

## Project layout

```
dataset/           train.csv, test.csv, sample_submission.csv
src/               Python pipeline (features, train, predict)
notebooks/         EDA and Colab-friendly training notebook
outputs/           models/ and submission.csv
```

## Local setup

```bash
cd Grid-Flipkart
pip install -r requirements.txt

# Recommended (v3 — ~94.5 OOF R², targets 94+ leaderboard)
python -m src.train_v3

# Legacy pipelines
python -m src.train      # original pipeline
python -m src.train_v2   # experimental
python -m src.predict    # only for legacy saved models
```

Submission file: `outputs/submission.csv` (41,778 rows).

### What changed in v3 (best)

- **5-fold KFold** averaged test predictions (same as `approaches/2ps.py`)
- **Raw demand target** + CatBoost `eval_metric=R2` (no log transform)
- **`timestamp` as categorical** + all 2ps time/geo features
- **`demand_d48`** — explicit day-48 same-slot demand (critical for day-49 test)
- **Optional hour calibration** blend from day-49 train ratios

### Optional hyperparameter tuning

```bash
python -m src.train --tune
```

## Google Colab

```python
# Cell 1: clone or upload project, then:
%cd /content/Grid-Flipkart   # adjust path
!pip install -q -r requirements.txt

# Cell 2: train
!python -m src.train

# Cell 3: predict
!python -m src.predict
```

Or open `notebooks/02_train_eval.ipynb` for an interactive run.

## Validation strategy

- **Holdout:** train on **day 48**, validate on **day 49** rows in train (0:00–2:00). Test covers day 49, 2:15–13:45 (not present in train labels).
- **Features:** day-48 same-slot demand, neighbor aggregates, target encodings, cyclical time.
- **Models:** CatBoost + LightGBM blend (log-target training).

Holdout scaled R² is typically ~50–65; test performance relies heavily on day-48 same-clock lags.

### Optional grid / CNN baseline

```python
from src.grid_model import train_grid_baseline, try_torch_cnn
# Day-48 grid lookup baseline or tiny Conv2D (requires torch on Colab GPU)
```

## Research basis

Optimized for small spatio-temporal panels (~77k rows, 2 days): gradient boosting + rich lags (Grab AI Traffic / ride-hailing demand literature), not full ST-GNN stacks that need PEMS-scale data.
