from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from scipy import stats
from .merge import merge
from .metrics import KS, AUC, F1, PSI

from .utils import (
    np_count,
    np_unique,
    to_ndarray,
    feature_splits,
    is_continuous,
    inter_feature,
    split_target,
)

from .utils.decorator import Decorator, support_dataframe

STATS_EMPTY = np.nan

def gini(target):
    """get gini index of a feature

    Args:
        target (array-like): list of target that will be calculate gini

    Returns:
        number: gini value
    """
    target = to_ndarray(target)
    v, c = np.unique(target, return_counts = True)

    return 1 - ((c / target.size) ** 2).sum()

def _gini_cond(feature, target):
    """private conditional gini function

    Args:
        feature (numpy.ndarray)
        target (numpy.ndarray)

    Returns:
        number: conditional gini value
    """
    size = feature.size

    value = 0
    for v, c in zip(*np_unique(feature, return_counts = True)):
        target_series = target[feature == v]
        value += c / size * gini(target_series)

    return value

@support_dataframe
def gini_cond(feature, target):
    """get conditional gini index of a feature

    Args:
        feature (array-like)
        target (array-like)

    Returns:
        number: conditional gini value. If feature is continuous, it will return the best gini value when the feature bins into two groups
    """
    if not is_continuous(feature):
        return _gini_cond(feature, target)

    # find best split for continuous data
    splits = feature_splits(feature, target)
    best = 999

    for f in inter_feature(feature, splits):
        v = _gini_cond(f, target)
        if v < best:
            best = v
    return best

def entropy(target):
    """get infomation entropy of a feature

    Args:
        target (array-like)

    Returns:
        number: information entropy
    """
    target = to_ndarray(target)
    uni, counts = np.unique(target, return_counts = True)
    prob = counts / len(target)
    entropy = stats.entropy(prob)
    return entropy

def _entropy_cond(feature, target):
    """private conditional entropy func

    Args:
        feature (numpy.ndarray)
        target (numpy.ndarray)

    Returns:
        number: conditional information entropy
    """
    size = len(feature)

    value = 0
    for v, c in zip(*np_unique(feature, return_counts = True)):
        target_series = target[feature == v]
        value += c/size * entropy(target_series)

    return value

@support_dataframe
def entropy_cond(feature, target):
    """get conditional entropy of a feature

    Args:
        feature (array-like)
        target (array-like)

    Returns:
        number: conditional information entropy. If feature is continuous, it will return the best entropy when the feature bins into two groups
    """
    feature = to_ndarray(feature)
    target = to_ndarray(target)

    if not is_continuous(feature):
        return _entropy_cond(feature, target)

    # find best split for continuous data
    splits = feature_splits(feature, target)
    best = 0
    for f in inter_feature(feature, splits):
        v = _entropy_cond(f, target)
        if v > best:
            best = v
    return best


def probability(target, mask = None):
    """get probability of target by mask
    """
    if mask is None:
        return 1, 1

    counts_0 = np_count(target, 0, default = 1)
    counts_1 = np_count(target, 1, default = 1)

    sub_target = target[mask]

    sub_0 = np_count(sub_target, 0, default = 1)
    sub_1 = np_count(sub_target, 1, default = 1)

    y_prob = sub_1 / counts_1
    n_prob = sub_0 / counts_0

    return y_prob, n_prob


def WOE(y_prob, n_prob):
    """get WOE of a group

    Args:
        y_prob: the probability of grouped y in total y
        n_prob: the probability of grouped n in total n

    Returns:
        number: woe value
    """
    return np.log(y_prob / n_prob)


def _IV(feature, target):
    """private information value func

    Args:
        feature (array-like)
        target (array-like)

    Returns:
        number: IV
        Series: IV of each groups
    """
    feature = to_ndarray(feature)
    target = to_ndarray(target)

    iv = {}

    for v in np.unique(feature):
        y_prob, n_prob = probability(target, mask = (feature == v))

        iv[v] = (y_prob - n_prob) * WOE(y_prob, n_prob)

    iv = pd.Series(iv)
    return iv.sum(), iv


@support_dataframe
def IV(feature, target, return_sub = False, **kwargs):
    """get the IV of a feature

    Args:
        feature (array-like)
        target (array-like)
        return_sub (bool): if need return IV of each groups
        n_bins (int): n groups that the feature will bin into
        method (str): the strategy to be used to merge feature, default is 'dt'
        **kwargs (): other options for merge function
    """
    if is_continuous(feature):
        feature = merge(feature, target, **kwargs)

    iv, sub = _IV(feature, target)

    if return_sub:
        return iv, sub
    
    return iv


def badrate(target):
    """calculate badrate

    Args:
        target (array-like): target array which `1` is bad

    Returns:
        float
    """
    return np.sum(target) / len(target)


def VIF(frame):
    """calculate vif

    Args:
        frame (ndarray|DataFrame)

    Returns:
        Series
    """
    index = None
    if isinstance(frame, pd.DataFrame):
        index = frame.columns
        frame = frame.values
    
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(fit_intercept = False)

    l = frame.shape[1]
    vif = np.zeros(l)

    for i in range(l):
        X = frame[:, np.arange(l) != i]
        y = frame[:, i]
        model.fit(X, y)

        pre_y = model.predict(X)

        vif[i] = np.sum(y ** 2) / np.sum((pre_y - y) ** 2)
    
    return pd.Series(vif, index = index)



class indicator(Decorator):
    """indicator decorator
    """
    # indicator name
    name = 'indicator'
    need_merge = False
    dtype = None

    def wrapper(self, *args, **kwargs):
        return self.call(*args, **kwargs)

# default indicators
INDICATORS = {
    'iv': indicator(name = 'iv', need_merge = True)(IV),
    'gini': indicator(name = 'gini')(gini_cond),
    'entropy': indicator(name = 'entropy')(entropy_cond),
    'auc': indicator(name = 'auc', dtype = np.number)(AUC),
    'ks': indicator(name = 'ks', dtype = np.number)(KS),
    'unique': indicator(name = 'unique')(lambda x, *arg: len(np_unique(x))),
}


def column_quality(feature, target, name = 'feature', indicators = [], need_merge = False, **kwargs):
    """calculate quality of a feature

    Args:
        feature (array-like)
        target (array-like)
        name (str): feature's name that will be setted in the returned Series
        indicators (list): list of indicator functions
        need_merge (bool): if need merge feature

    Returns:
        Series: a list of quality with the feature's name
    """
    feature = to_ndarray(feature)
    target = to_ndarray(target)

    if not np.issubdtype(feature.dtype, np.number):
        feature = feature.astype(str)

    if need_merge:
        bin_feature = merge(feature, target, **kwargs)

    res = {}  
    for func in indicators:
        # filter by dtype
        if func.dtype is not None and not isinstance(feature.dtype, func.dtype):
            res[func.name] = STATS_EMPTY
            continue

        # if function need use bin feature
        if func.need_merge:
            res[func.name] = func(bin_feature, target)
            continue
        
        res[func.name] = func(feature, target)

    row = pd.Series(res)

    row.name = name
    return row


def quality(dataframe, target = 'target', cpu_cores = 0, iv_only = False, indicators = ['iv', 'gini', 'entropy', 'unique'], **kwargs):
    """get quality of features in data

    Args:
        dataframe (DataFrame): dataframe that will be calculate quality
        target (str): the target's name in dataframe
        iv_only (bool): `deprecated`. if only calculate IV
        cpu_cores (int): the maximun number of CPU cores will be used, `0` means all CPUs will be used, 
            `-1` means all CPUs but one will be used.

    Returns:
        DataFrame: quality of features with the features' name as row name
    """
    frame, target = split_target(dataframe, target)

    if iv_only:
        import warnings
        warnings.warn(
            """`iv_only` will be deprecated soon,
                please use `indicators = ['iv']` instead!
            """,
            DeprecationWarning,
        )

        dummy_func = lambda x, t: STATS_EMPTY

        indicators = [
            'iv',
            indicator(name = 'gini')(dummy_func),
            indicator(name = 'entropy')(dummy_func),
            'unique',
        ]
    
    
    need_merge = False
    for i, f in enumerate(indicators):
        # replace str type indicator to function
        if isinstance(f, str):
            assert f in INDICATORS

            indicators[i] = INDICATORS[f]
        
        # update need merge flag
        need_merge |= indicators[i].need_merge
    
    
    if cpu_cores < 1:
        cpu_cores = cpu_cores - 1
    
    
    pool = Parallel(n_jobs = cpu_cores)

    jobs = []
    for name, series in frame.iteritems():
        jobs.append(delayed(column_quality)(
            series,
            target,
            name = name,
            indicators = indicators,
            need_merge = need_merge,
            **kwargs
        ))

    rows = pool(jobs)


    return pd.DataFrame(rows).sort_values(
        by = indicators[0].name,
        ascending = False,
    )
