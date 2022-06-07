import numpy as np
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.datasets import fetch_openml, make_classification
from time import time

def get_openml(name):
    name_to_id = {
        "higgs": 23512,
        "miniboone": 41150,
        "christine": 41142,
        "volkert": 41166,
        "wine": 40498,
        "phoneme": 1489,
        "dilbert": 41163
    }

    data_id = name_to_id[name]

    data = fetch_openml(data_id=data_id)
    X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
    y = LabelEncoder().fit_transform(y)

    # Dataset specific preprocessing
    if name == "higgs":
        X = X.iloc[:-1, :] # last columns has NAs
        y = y[:-1]
    elif name == "christine" or name == "volkert":
        X = X[[c for c in X.columns if X[c].nunique() > 2]] # Remove useless features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    return X_train, X_test, y_train, y_test

def make_classifier(n, p, k, seed=10):
    rng = np.random.default_rng(seed)
    if p > k:
        A = rng.normal(0, 1, size=(k, k))
        cov = A.T @ A
        X = rng.multivariate_normal(mean=rng.normal(size=k), cov=cov, size=n)

        n_extra_feats = p - k
        W = rng.normal(size=(k, n_extra_feats))
        X = np.column_stack([X, X @ W])
    else:
        A = rng.normal(0, 1, size=(p // 2, p))
        cov = A.T @ A
        X = rng.multivariate_normal(mean=rng.normal(size=p), cov=cov, size=n)

    beta = rng.normal(size=p)
    logits = X @ beta + rng.normal(loc=0, scale=5, size=n)
    logits = scale(logits.reshape(-1, 1)).reshape(-1)
    y = rng.binomial(n=1, p = 1 / (1 + np.exp(-2*logits)), size=n)

    return X, y

def time_fit(model, X_train, y_train):
    start = time()
    model.fit(X_train, y_train)
    end = time()
    return end - start

def make_val_split(X_train, seed=10, val_size=0.2):
    X_train_, _ = train_test_split(X_train, test_size=val_size, random_state=seed)

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if i in X_train_.index else 0 for i in X_train.index]
    pds = PredefinedSplit(test_fold = split_index)
    return pds

def get_data(args):
    print('Creating datasets and dataloaders')
    if args.data == 'synthetic':
        print(f'Using synthetic data with seed {args.set_seed}')
        X, y = make_classification(n_samples=args.num_samples, n_features=args.num_features, n_informative=4, 
                                    n_redundant=0, n_clusters_per_class=2, random_state=args.set_seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

        return X_train, X_test, y_train, y_test
    
    elif args.data == 'openml':
        print(f'Using openML data: {args.dset_name}')
        X_train, X_test, y_train, y_test = get_openml(args.dset_name)
        
        return X_train, X_test, y_train, y_test
    
    else:
        raise NotImplementedError
    