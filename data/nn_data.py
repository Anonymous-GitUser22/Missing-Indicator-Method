import numpy as np
import torch
from torch.utils.data import Dataset

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.imputation_utils import simple_mask, MNAR_mask

class DataSet(Dataset):
    def __init__(self, X, y, miss_mech, seed, mcar_p, mnar_gamma, scalar, train=True):
        self.miss_mech = miss_mech
        self.scalar = scalar
        if self.miss_mech == 'mcar':
            if train:
                print('Using mcar as the missing mechanism')
                print(f'Using missingness of {mcar_p}')
            # MCAR Mask
            self.mcar_p = mcar_p
            self.mask = simple_mask(X, p=self.mcar_p, seed=seed, return_na=True)
        
        if self.miss_mech == 'mnar':
            if train:
                print('Using mnar as the missing mechanism')
                print(f'Using missingness of {mnar_gamma}')
            # MNAR Mask
            self.mnar_gamma = mnar_gamma
            self.probs, self.mask = MNAR_mask(X, side="right", power=self.mnar_gamma, seed=seed, return_na=True, standardize=True)

        self.X_masked_nan = X * self.mask
        # scalar transform here
        if train:
            self.X_masked_nan = self.scalar.fit_transform(self.X_masked_nan)
        else:
            # Must be run after train
            self.X_masked_nan = self.scalar.transform(self.X_masked_nan)
        # replace nans with zeros
        self.X_masked = np.nan_to_num(self.X_masked_nan, copy=True, nan=0.0)
        self.mask_feats = np.where(np.isnan(self.mask), 1, 0)
        self.X = np.column_stack([self.X_masked, self.mask_feats])
        if len(y.shape)==1:
            self.y = y[:,None]
        else:
            self.y = y
        self.cls = np.zeros_like(self.y,dtype=int)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.cls[idx], self.X[idx], self.y[idx]

