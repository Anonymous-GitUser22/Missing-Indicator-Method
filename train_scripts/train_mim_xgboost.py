import numpy as np
from time import time
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nn_data import DataSet
from utils.tabular_utils import get_data

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--savemodelroot', default='./output', type=str)
parser.add_argument('--run_name', default='xgboost', type=str)
parser.add_argument('--seed', default=10, type=int)

parser.add_argument('--data', default='openml', type=str, choices = ['openml','synthetic'])
parser.add_argument('--dset_name', type=str)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--num_features', default=4, type=int)

parser.add_argument('--miss_mech', default='mnar', type=str, choices = ['mcar','mnar'])
parser.add_argument('--mcar_p', default=0.5, type=float)
parser.add_argument('--mnar_gamma', default=1., type=float)

parser.add_argument('--n_estimators', default=100, type=int)
parser.add_argument('--max_depth', default=6, type=int)

args = parser.parse_args()

if args.data == 'openml':
    modelsave_path = os.path.join(os.getcwd(),args.savemodelroot,args.data,args.dset_name,args.run_name)
elif args.data == 'synthetic':
    modelsave_path = os.path.join(os.getcwd(),args.savemodelroot,args.data+'_'+str(args.seed),args.run_name)
os.makedirs(modelsave_path, exist_ok=True)

X_train, X_test, y_train, y_test = get_data(args)

scalar = StandardScaler()
train_ds = DataSet(X_train, y_train, args.miss_mech, args.seed, args.mcar_p, args.mnar_gamma, scalar, train=True)
X_train, y_train = train_ds.X, (train_ds.y).ravel()

test_ds = DataSet(X_test, y_test, args.miss_mech, args.seed, args.mcar_p, args.mnar_gamma, train_ds.scalar, train=False)
X_test, y_test = test_ds.X, (test_ds.y).ravel()

model = xgboost.XGBClassifier(n_jobs=-1, seed=args.seed, n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr)

model.fit(X_train,y_train)

acc = accuracy_score(y_test, model.predict(X_test))
params = model.get_xgb_params()

# model.save_model(os.path.join(modelsave_path,f'last_model_{args.seed}.json'))

# with open(os.path.join(modelsave_path,f'log_{args.seed}.txt'), "w") as f:
#     f.write(f'Accuracy: {acc}')
#     f.write('\n'+str(params))

print('Final ACCURACY:  %.3f' %(acc*100))
    