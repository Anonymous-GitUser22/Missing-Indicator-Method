import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import argparse
import os
import sys
sys.path.append('../')
from utils.imputation_utils import simple_mask
from models.imputation_models import LRGCImputer
from utils.tabular_utils import make_classifier, time_fit

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="logreg", type=str)
parser.add_argument("--n_trials", default=5, type=int)
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--n_jobs", default=1, type=int)
args = parser.parse_args()

seed = args.seed
n = 10000
k = 10
n_feats = [int(x) for x in np.linspace(15, 500, 20)]

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

results = []
first_seed = seed

model_params["n_jobs"] = args.n_jobs

for p in n_feats:

    print(p)

    seed = first_seed

    for _ in range(args.n_trials):
        seed += 1
        # Set seed for model
        model_params["random_state"] = seed

        X, y = make_classifier(n, p, k, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

        # MCAR Mask
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


        #Impute with Low Rank GC
        gc_model = make_pipeline(
            StandardScaler(),
            LRGCImputer(rank=k, random_state=seed, verbose=2),
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

        indicator_gc_model = make_pipeline(
            make_column_transformer(
                (
                    make_pipeline(StandardScaler(), LRGCImputer(rank=k, random_state=seed, verbose=2)), 
                    np.arange(0, X.shape[1])
                ),
                remainder="passthrough"
            ),
            model(**model_params)
        )

        model_scores = [0]*5
        model_times = [0]*5

        baseline_time = time_fit(baseline, X_train, y_train)
        model_scores[0] = accuracy_score(y_test, baseline.predict(X_test))
        model_times[0] = baseline_time

        mean_model_time = time_fit(mean_model, X_train_masked, y_train)
        model_scores[1] = accuracy_score(y_test, mean_model.predict(X_test_masked))
        model_times[1] = mean_model_time

        gc_model_time = time_fit(gc_model, X_train_masked, y_train)
        model_scores[2] = accuracy_score(y_test, gc_model.predict(X_test_masked))
        model_times[2] = gc_model_time

        indicator_model_time = time_fit(
            indicator_model, np.column_stack([X_train_masked, train_mask_feats]), y_train
        )
        model_scores[3] = accuracy_score(
            y_test, indicator_model.predict(np.column_stack([X_test_masked, test_mask_feats]))
        )
        model_times[3] = indicator_model_time

        indicator_gc_model_time = time_fit(
            indicator_gc_model, np.column_stack([X_train_masked, train_mask_feats]), y_train
        )
        model_scores[4] = accuracy_score(
            y_test, indicator_gc_model.predict(np.column_stack([X_test_masked, test_mask_feats]))
        )
        model_times[4] = indicator_gc_model_time
        
        results.append([p] + model_scores + model_times)
    
        print(model_scores)
        print(model_times)   

results_df = pd.DataFrame(results, columns=["NumFeatures", "Oracle", "Mean", "GC", "Mean_Indicator","GC_Indicator",
                                    "Oracle_Time", "Mean_Time", "GC_Time", "Mean_Indicator_Time", "GC_Indicator_Time"])

print(results_df)

if not os.path.exists("outputs"):
    os.makedirs("outputs")

results_df.to_csv("outputs/sim_by_p.csv", index=False)
