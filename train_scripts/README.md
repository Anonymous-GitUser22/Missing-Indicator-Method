# Missing Indicator Method Training Scripts

This directory contains scripts to reproduce the data used to recreate the data used to make the figures and tables in this paper.  Here is a mapping of script to table/figure:

Figure 2 -> simulated_data_by_gamma.py
Figure 3 -> simulated_data_by_p.py
Table 1,2 -> train_preprocessor_experiment.py
Table S4 -> train_mim_*.py

### Arguments for simulated_data_by_gamma.py and simulated_data_by_p.py

`--model`: Model used (default: logreg; choices: logreg, xgboost, mlp).

`--n_trials`: Number of trials to run each configuration (default: 5).

`--seed`: Seed for controlling randomness (default: 10).

`--n_jobs`: Number of (sklearn) parallel jobs to run (default: 1).

### Arguments for train_preprocessor_experiment.py

`--dataset`: OpenML dataset to run experiment on (choices: "higgs", "miniboone", "christine", "volkert", "wine", "phoneme", "dilbert").

`--gc_type`: Type of gcimpute model to use (default: "gc"; choices: "gc", "lrgc").

`--gamma`: Informativeness parameter for mask generation (default: 1)

`--n_trials`: Number of trials to run each configuration (default: 5).

`--seed`: Seed for controlling randomness (default: 10).

## Arguments for all train_mim_* scripts

`--lr`: learning rate.

`--savemodelroot`: Name of root directory to save output files (default: "./output").

`--run_name`: Name of the run, will be used to distinguish this specific run in the output directory.

`--seed`: Seed for controlling randomness (default: 10).

`--data`: Type of data (default: "openml"; choices: "openml", "synthetic").

`--data_name`: Name of openml dataset (choices: "higgs", "miniboone", "christine", "volkert", "wine", "phoneme", "dilbert").

`--num_samples`: Number of generated syntehtic samples (default: 10000).

`--num_features`: Number of generated syntehtic features (default: 4).

`--miss_mech`: Impuatation type (choices: "mcar", "mnar").

`--mcar_p`: Mcar probability value (default: 0.5).

`--mnar_gamma`: MNAR gamma (default: 1.0).

### Arguments for both train_mim_trasformer.py and train_mim_mlp.py

`--epochs`: training epochs (default: 100).

`--batchsize`: training minbatch size (default: 256).

`--optimizer`: type of optimizer (default : "AdamW"; choices: "AdamW" | "Adam" | "SGD").

`--scheduler`: SGD scheduler, if used  (default: cosine; choices: "cosine" | "linear").

### Arguments for train_mim_transformer.py

`--embedding_size`: dimension of the embedding of CLS token and tabluar data (default: 8).

`--transformer_depth`: Number of transformer encoder blocks (default: 1).

`--attention_heads`: Number of heads for multiheadattention (default: 8).

`--viz`: Flag indicating if attention weights should be saved during forward pass (default: False).

### Arguments for train_mim_mlp.py

`--mlp_layers`: dimensions of the hidden mlp layers, space between each dimension. Can use either raw numbers or multiples of data dimension, p (default: "8p" "4p").

### Arguments for train_mim_xgboost.py

`--n_estimators`: Number of xgboost boosting rounds (default: 100).

`--max_depth`: Maximum xgboost tree depth for base learners (default: 6).
