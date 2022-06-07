import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tqdm import tqdm

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import argparse
import os
import sys
sys.path.append('../')
from utils.imputation_utils import MNAR_mask
from models.imputation_models import GCImputer
from utils.tabular_utils import time_fit

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="logreg", type=str)
parser.add_argument("--n_trials", default=5, type=int)
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--n_jobs", default=1, type=int)
args = parser.parse_args()


seed = args.seed
X, y = make_classification(n_samples=10000, n_features=4, n_informative=4, 
                           n_redundant=0, n_clusters_per_class=2, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

if args.model == "logreg":
    model = LogisticRegression
    model_params = {"penalty": "none"} 
elif args.model == "xgboost":
    model = XGBClassifier
    model_params = {"use_label_encoder": False}
elif args.model == "mlp":
    model = MLPClassifier
    model_params = {
        "hidden_layer_sizes": (16, 16),
        "alpha": 0,
        "batch_size": 128, 
        "max_iter": 30, 
        "solver":"adam"
    }
else:
    raise ValueError("model must be one of logreg, xgboost, mlp.")

model_params["n_jobs"] = args.n_jobs

gammas = np.linspace(0, 5, 25)
results = []

for i in range(args.n_trials):
    
    seed += 1

    for power in tqdm(gammas):
        
        # Set seed for model
        model_params["random_state"] = seed

        # MNAR Mask
        train_probs, train_mask = MNAR_mask(X_train, side="right", power=power, seed=seed, return_na=True, standardize=True)
        X_train_masked = X_train * train_mask
        train_mask_feats = np.where(np.isnan(train_mask), 1, 0)

        test_probs, test_mask = MNAR_mask(X_test, side="right", power=power, seed=seed, return_na=True, standardize=True)
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
        
        # Impute with MissForest
        mf_model = make_pipeline(
            StandardScaler(),
            IterativeImputer(estimator=RandomForestRegressor(n_jobs=args.n_jobs, random_state=seed)),
            model(**model_params)
        )

        gc_model = make_pipeline(
            StandardScaler(),
            GCImputer(random_state=seed, n_jobs=args.n_jobs),
            model(**model_params)
        )


        # Impute with 0 and add indicators
        indicator_model = make_pipeline(
            make_column_transformer(
                (
                    make_pipeline(StandardScaler(), SimpleImputer(strategy="constant", fill_value=0)), 
                    np.arange(0, X.shape[1])
                ),
                remainder="passthrough"
            ),
            model(**model_params)
        )

        # Impute with MF and add indicators
        indicator_mf_model = make_pipeline(
            make_column_transformer(
                (
                    make_pipeline(StandardScaler(), IterativeImputer(estimator=RandomForestRegressor(random_state=seed, n_jobs=args.n_jobs))), 
                    np.arange(0, X.shape[1])
                ),
                remainder="passthrough"
            ),
            model(**model_params)
        )

        # Impute with GC and add indicators
        indicator_gc_model = make_pipeline(
            make_column_transformer(
                (
                    make_pipeline(StandardScaler(), GCImputer(random_state=seed, n_jobs=args.n_jobs)), 
                    np.arange(0, X.shape[1])
                ),
                remainder="passthrough"
            ),
            model(**model_params)
        )

        model_scores = [0]*7
        model_times = [0]*7

        baseline_time = time_fit(baseline, X_train, y_train)
        model_scores[0] = accuracy_score(y_test, baseline.predict(X_test))
        model_times[0] = baseline_time

        mean_model_time = time_fit(mean_model, X_train_masked, y_train)
        model_scores[1] = accuracy_score(y_test, mean_model.predict(X_test_masked))
        model_times[1] = mean_model_time
        
        mf_model_time = time_fit(mf_model, X_train_masked, y_train)
        model_scores[2] = accuracy_score(y_test, mf_model.predict(X_test_masked))
        model_times[2] = mf_model_time

        gc_model_time = time_fit(gc_model, X_train_masked, y_train)
        model_scores[3] = accuracy_score(y_test, gc_model.predict(X_test_masked))
        model_times[3] = gc_model_time

        indicator_model_time = time_fit(
            indicator_model, np.column_stack([X_train_masked, train_mask_feats]), y_train
        )
        model_scores[4] = accuracy_score(
            y_test, indicator_model.predict(np.column_stack([X_test_masked, test_mask_feats]))
        )
        model_times[4] = indicator_model_time

        indicator_mf_model_time = time_fit(
            indicator_mf_model, np.column_stack([X_train_masked, train_mask_feats]), y_train
        )
        model_scores[5] = accuracy_score(
            y_test, indicator_mf_model.predict(np.column_stack([X_test_masked, test_mask_feats]))
        )
        model_times[5] = indicator_mf_model_time

        indicator_gc_model_time = time_fit(
            indicator_gc_model, np.column_stack([X_train_masked, train_mask_feats]), y_train
        )
        model_scores[6] = accuracy_score(
            y_test, indicator_gc_model.predict(np.column_stack([X_test_masked, test_mask_feats]))
        )
        model_times[6] = indicator_gc_model_time

        results.append([power] + model_scores + model_times)
    
results_df = pd.DataFrame(results, columns=["Power", "Oracle", "Mean", "MF", "GC", "Mean_Indicator", "MF_Indicator", "GC_Indicator",
                                        "Oracle_Time", "Mean_Time", "MF_Time", "GC_Time", "Mean_Indicator_Time", "MF_Indicator_Time", "GC_Indicator_Time"])

print(results_df)

if not os.path.exists("outputs"):
    os.makedirs("outputs")
results_df.to_csv("outputs/sim_by_gamma.csv", index=False)
