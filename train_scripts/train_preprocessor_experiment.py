import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import argparse
import os
import sys
sys.path.append('../')
from utils.imputation_utils import simple_mask, MNAR_mask
from models.imputation_models import GCImputer, LRGCImputer
from utils.tabular_utils import time_fit, make_val_split, get_openml


# This is needed to ignore warnings produced by Logistic Regression model.
# The model is run for a maximum of 1000 iterations, even if sklearn does not deem the model to have converged.
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = 'ignore::ConvergenceWarning:sklearn.model_selection.GridSearchCV'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--gc_type", default="gc", type=str)
parser.add_argument('--gamma', default=1., type=float)  # Use gamma=0 for MCAR mask
parser.add_argument("--n_trials", default=5, type=int)
parser.add_argument("--seed", default=10, type=int)
args = parser.parse_args()

X_train, X_test, y_train, y_test = get_openml(args.dataset)
val_split = make_val_split(X_train, seed=10, val_size=0.2)

seed = args.seed
model = LogisticRegression
model_params = {"penalty": "none", "max_iter": 1000}

# Check gc type
assert args.gc_type in ["gc", "lrgc"], "gc_type must be gc or lrgc"

results = []

for _ in tqdm(range(args.n_trials)):
    
    seed += 1
    print(seed)
        
    # Set seed for model
    model_params["random_state"] = seed

    # Mask data
    if args.gamma > 0:
        _, train_mask = MNAR_mask(X_train, side="right", power=args.gamma, seed=seed, return_na=True, standardize=True)
        X_train_masked = X_train * train_mask
        train_mask_feats = np.where(np.isnan(train_mask), 1, 0)

        _, test_mask = MNAR_mask(X_test, side="right", power=args.gamma, seed=seed, return_na=True, standardize=True)
        X_test_masked = X_test * test_mask
        test_mask_feats = np.where(np.isnan(test_mask), 1, 0)
    else:
        train_mask = simple_mask(X_train, p=0.5, seed=seed, return_na=True)
        X_train_masked = X_train * train_mask
        train_mask_feats = np.where(np.isnan(train_mask), 1, 0)

        test_mask = simple_mask(X_test, p=0.5, seed=seed, return_na=True)
        X_test_masked = X_test * test_mask
        test_mask_feats = np.where(np.isnan(test_mask), 1, 0)

    # Baseline
    baseline = make_pipeline(
        StandardScaler(),
        model(**model_params)
    )

    # Impute with 0
    mean_model = make_pipeline(
        StandardScaler(),
        SimpleImputer(strategy="constant", fill_value=0),
        model(**model_params)
    )

    # Impute with GC
    if args.gc_type == "gc":
        gc_model = make_pipeline(
            StandardScaler(),
            GCImputer(random_state=seed, min_ord_ratio=np.inf, n_jobs=1, max_iter=50, verbose=2),
            model(**model_params)
        )
    else:
        gc_model_base = make_pipeline(
            StandardScaler(),
            LRGCImputer(rank=10, random_state=seed, min_ord_ratio=np.inf, n_jobs=1, max_iter=50, verbose=2),
            model(**model_params)
        )
        gc_model = GridSearchCV(
            estimator=gc_model_base,
            cv=val_split,
            param_grid={"lrgcimputer__rank": [5, 10, 15, 20]},
            n_jobs=4
        )

    # Impute with 0 and add indicators
    indicator_model = make_pipeline(
        make_column_transformer(
            (
                make_pipeline(StandardScaler(), SimpleImputer(strategy="constant", fill_value=0)), 
                np.arange(0, X_train.shape[1])
            ),
            remainder="passthrough"
        ),
        model(**model_params)
    )

    # Impute with GC and add indicators
    if args.gc_type == "gc":
        indicator_gc_model = make_pipeline(
            make_column_transformer(
                (
                    make_pipeline(StandardScaler(), GCImputer(random_state=seed, min_ord_ratio=np.inf, n_jobs=1, max_iter=50, verbose=2)), 
                    np.arange(0, X_train.shape[1])
                ),
                remainder="passthrough"
            ),
            model(**model_params)
        )
    else:
        indicator_gc_model_base = make_pipeline(
            make_column_transformer(
                (
                    make_pipeline(StandardScaler(), LRGCImputer(rank=10, random_state=seed, min_ord_ratio=np.inf, n_jobs=1, max_iter=50, verbose=2)), 
                    np.arange(0, X_train.shape[1])
                ),
                remainder="passthrough"
            ),
            model(**model_params)
        )
        indicator_gc_model = GridSearchCV(
            estimator=indicator_gc_model_base,
            cv=val_split,
            param_grid={"lrgcimputer__rank": [5, 10, 15, 20]},
            n_jobs=4
        )

    model_scores = [0]*5
    model_times = [0]*5

    print("Oracle")
    baseline_time = time_fit(baseline, X_train, y_train)
    model_scores[0] = accuracy_score(y_test, baseline.predict(X_test))
    model_times[0] = baseline_time

    print("Mean")
    mean_model_time = time_fit(mean_model, X_train_masked, y_train)
    model_scores[1] = accuracy_score(y_test, mean_model.predict(X_test_masked))
    model_times[1] = mean_model_time
    
    print(model_scores)

    print("GC")
    gc_model_time = time_fit(gc_model, X_train_masked, y_train)
    model_scores[2] = accuracy_score(y_test, gc_model.predict(X_test_masked))
    model_times[2] = gc_model_time

    print("Mean + MIM")
    indicator_model_time = time_fit(
        indicator_model, np.column_stack([X_train_masked, train_mask_feats]), y_train
    )
    model_scores[3] = accuracy_score(
        y_test, indicator_model.predict(np.column_stack([X_test_masked, test_mask_feats]))
    )
    model_times[3] = indicator_model_time

    print("GC + MIM")
    indicator_gc_model_time = time_fit(
        indicator_gc_model, np.column_stack([X_train_masked, train_mask_feats]), y_train
    )
    model_scores[4] = accuracy_score(
        y_test, indicator_gc_model.predict(np.column_stack([X_test_masked, test_mask_feats]))
    )
    model_times[4] = indicator_gc_model_time

    results.append(model_scores + model_times)
    print(model_scores)
    print(model_times)

results_df = pd.DataFrame(results, columns=["Oracle", "Mean", "GC", "Mean_Indicator","GC_Indicator",
                                    "Oracle_Time", "Mean_Time", "GC_Time", "Mean_Indicator_Time", "GC_Indicator_Time"])

print(results_df)

if not os.path.exists("outputs"):
    os.makedirs("outputs")

if args.gamma == 0:
    results_df.to_csv(f"outputs/results_mcar_{args.dataset}.csv", index=False)
else:
    results_df.to_csv(f"outputs/results_mnar_{args.dataset}.csv", index=False)




