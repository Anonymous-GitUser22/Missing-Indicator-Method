import numpy as np
from sklearn.preprocessing import StandardScaler

def simple_mask(X, p=0.5, rng=None, seed=None, return_na=False):
    if not rng and not seed:
        rng = np.random.default_rng()
    elif not rng:
        rng = np.random.default_rng(seed)

    # Simple MCAR mask
    mask = rng.binomial(n=1, p=p, size=X.shape)

    if return_na:
        mask = np.where(mask == 0, np.nan, mask)

    return mask

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MNAR_mask(x, side="tail", rng=None, seed=None, power=1, standardize=False, return_na=False):
    # if not isinstance(x, np.ndarray):
    #     x = x.to_numpy("float")

    if not rng and not seed:
        rng = np.random.default_rng()
    elif not rng:
        rng = np.random.default_rng(seed)

    if standardize:
        x = StandardScaler().fit_transform(x)

    if side == "tail":
        probs = sigmoid((np.abs(x) - 0.75)*power)
    elif side == "mid":
        probs = sigmoid((-np.abs(x) + 0.75)*power)
    elif side == "left":
        probs = sigmoid(-x*power)
    elif side == "right":
        probs = sigmoid(x*power)
    else:
        raise ValueError(f"Side must be one of tail, mid, left, or right, got {side}")

    mask = rng.binomial(1, probs, size=x.shape)
    
    if return_na:
        mask = np.where(mask == 0, np.nan, mask)
    
    return probs, mask
