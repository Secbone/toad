import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .merge import merge

from .utils import (
    feature_splits,
    iter_df,
    unpack_tuple,
    bin_by_splits,
)


def KS(score, target):
    """calculate ks value

    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target

    Returns:
        float: the max KS value
    """
    mask = target == 1
    res = ks_2samp(score[mask], score[~mask])
    return res[0]


def KS_bucket(score, target, bucket = 10, method = 'quantile', return_splits = False, **kwargs):
    """calculate ks value by bucket

    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target
        bucket (int): n groups that will bin into
        method (str): method to bin score. `quantile` (default), `step`
        return_splits (bool): if need to return splits of bucket

    Returns:
        DataFrame
    """
    df = pd.DataFrame({
        'score': score,
        'bad': target,
    })

    df['good'] = 1 - df['bad']

    bad_total = df['bad'].sum()
    good_total = df['good'].sum()
    all_total = bad_total + good_total

    splits = None
    df['bucket'] = 0

    if bucket is False:
        df['bucket'] = score
    elif isinstance(bucket, (list, np.ndarray, pd.Series)):
        # list of split pointers
        if len(bucket) < len(score):
            bucket = bin_by_splits(score, bucket)
        
        df['bucket'] = bucket
    elif isinstance(bucket, int):
        df['bucket'], splits = merge(score, n_bins = bucket, method = method, return_splits = True, **kwargs)

    grouped = df.groupby('bucket', as_index = False)

    agg1 = pd.DataFrame()
    agg1['min'] = grouped.min()['score']
    agg1['max'] = grouped.max()['score']
    agg1['bads'] = grouped.sum()['bad']
    agg1['goods'] = grouped.sum()['good']
    agg1['total'] = agg1['bads'] + agg1['goods']

    agg2 = (agg1.sort_values(by = 'min')).reset_index(drop = True)

    agg2['bad_rate'] = agg2['bads'] / agg2['total']
    agg2['good_rate'] = agg2['goods'] / agg2['total']

    agg2['odds'] = agg2['bads'] / agg2['goods']

    agg2['bad_prop'] = agg2['bads'] / bad_total
    agg2['good_prop'] = agg2['goods'] / good_total
    agg2['total_prop'] = agg2['total'] / all_total
    

    cum_bads = agg2['bads'].cumsum()
    cum_goods = agg2['goods'].cumsum()
    cum_total = agg2['total'].cumsum()

    cum_bads_rev = agg2.loc[::-1, 'bads'].cumsum()[::-1]
    cum_goods_rev = agg2.loc[::-1, 'goods'].cumsum()[::-1]
    cum_total_rev = agg2.loc[::-1, 'total'].cumsum()[::-1]

    agg2['cum_bad_rate'] = cum_bads / cum_total
    agg2['cum_bad_rate_rev'] = cum_bads_rev / cum_total_rev
    
    agg2['cum_bads_prop'] = cum_bads / bad_total
    agg2['cum_bads_prop_rev'] = cum_bads_rev / bad_total
    agg2['cum_goods_prop'] = cum_goods / good_total
    agg2['cum_goods_prop_rev'] = cum_goods_rev / good_total
    agg2['cum_total_prop'] = cum_total / all_total
    agg2['cum_total_prop_rev'] = cum_total_rev / all_total


    agg2['ks'] = agg2['cum_bads_prop'] - agg2['cum_goods_prop']

    reverse_suffix = ''
    # fix negative ks value
    if agg2['ks'].sum() < 0:
        agg2['ks'] = -agg2['ks']
        reverse_suffix = '_rev'
    
    agg2['lift'] = agg2['cum_bads_prop' + reverse_suffix] / agg2['cum_total_prop' + reverse_suffix]

    if return_splits and splits is not None:
        return agg2, splits
    
    return agg2

def KS_by_col(df, by='feature', score='score', target='target'):
    """
    """

    pass


def SSE(y_pred, y):
    """sum of squares due to error
    """
    return np.sum((y_pred - y) ** 2)


def MSE(y_pred, y):
    """mean of squares due to error
    """
    return np.mean((y_pred - y) ** 2)


def AIC(y_pred, y, k, llf = None):
    """Akaike Information Criterion

    Args:
        y_pred (array-like)
        y (array-like)
        k (int): number of featuers
        llf (float): result of log-likelihood function
    """
    if llf is None:
        llf = np.log(SSE(y_pred, y))

    return 2 * k - 2 * llf


def BIC(y_pred, y, k, llf = None):
    """Bayesian Information Criterion

    Args:
        y_pred (array-like)
        y (array-like)
        k (int): number of featuers
        llf (float): result of log-likelihood function
    """
    n = len(y)
    if llf is None:
        llf = np.log(SSE(y_pred, y))

    return np.log(n) * k - 2 * llf


def F1(score, target, split = 'best', return_split = False):
    """calculate f1 value

    Args:
        score (array-like)
        target (array-like)

    Returns:
        float: best f1 score
        float: best spliter
    """
    dataframe = pd.DataFrame({
        'score': score,
        'target': target,
    })

    if split == 'best':
        # find best split for score
        splits = feature_splits(dataframe['score'], dataframe['target'])
    else:
        splits = [split]

    best = 0
    sp = None
    for df, pointer in iter_df(dataframe, 'score', 'target', splits):
        v = f1_score(df['target'], df['score'])

        if v > best:
            best = v
            sp = pointer

    if return_split:
        return best, sp

    return best


def AUC(score, target, return_curve = False):
    """AUC Score

    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target
        return_curve (bool): if need return curve data for ROC plot

    Returns:
        float: auc score
    """
    # fix score order
    if np.nanmax(score) > 1:
        score = -score

    auc = roc_auc_score(target, score)

    if not return_curve:
        return auc
    
    return (auc,) + roc_curve(target, score)


def _PSI(test, base):
    test_prop = pd.Series(test).value_counts(normalize = True, dropna = False)
    base_prop = pd.Series(base).value_counts(normalize = True, dropna = False)

    psi = np.sum((test_prop - base_prop) * np.log(test_prop / base_prop))

    frame = pd.DataFrame({
        'test': test_prop,
        'base': base_prop,
    })
    frame.index.name = 'value'

    return psi, frame.reset_index()



def PSI(test, base, combiner = None, return_frame = False):
    """calculate PSI

    Args:
        test (array-like): data to test PSI
        base (array-like): base data for calculate PSI
        combiner (Combiner|list|dict): combiner to combine data
        return_frame (bool): if need to return frame of proportion

    Returns:
        float|Series
    """

    if combiner is not None:
        if isinstance(combiner, (dict, list)):
            from .transform import Combiner
            combiner = Combiner().load(combiner)

        test = combiner.transform(test, labels = True)
        base = combiner.transform(base, labels = True)

    psi = list()
    frame = list()

    if isinstance(test, pd.DataFrame):
        for col in test:
            p, f = _PSI(test[col], base[col])
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index = test.columns)

        frame = pd.concat(
            frame,
            keys = test.columns,
            names = ['columns', 'id'],
        ).reset_index()
        frame = frame.drop(columns = 'id')
    else:
        psi, frame = _PSI(test, base)


    res = (psi,)

    if return_frame:
        res += (frame,)

    return unpack_tuple(res)


def matrix(y_pred, y, splits = None):
    """confusion matrix of target

    Args:
        y_pred (array-like)
        y (array-like)
        splits (float|list): split points of y_pred

    Returns:
        DataFrame: confusion matrix witch true labels in rows and predicted labels in columns

    """
    if splits is not None:
        y_pred = bin_by_splits(y_pred, splits)
    
    labels = np.unique(y)
    from sklearn.metrics import confusion_matrix
    m = confusion_matrix(y, y_pred, labels = labels)

    return pd.DataFrame(
        m,
        index = pd.Index(labels, name = 'Actual'),
        columns = pd.Index(labels, name = 'Predicted'),
    )