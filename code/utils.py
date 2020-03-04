import pandas as pd
import numpy as np
import scipy.sparse as sparse

from implicit.nearest_neighbours import bm25_weight, tfidf_weight, normalize


def make_matrix(df, count=False):
    """
    Args:
        plu_column: string. Name of column which contains PLU
        card_column: string. Name of column which contains Card numbers
    Returns:
        df: dataframe
    """
    if count:
        return df \
            .pivot_table(index=['user_id'], columns=['item_id'], values='feedback', aggfunc='count') \
            .reset_index(drop=True) \
            .fillna(0)

    else:
        return df \
            .pivot_table(index=['user_id'], columns=['item_id'], values='feedback') \
            .reset_index(drop=True) \
            .fillna(0)


def transform(matrix, method='no', clip_upper_value=100):
    """
    Function transforms every single value in matrix with specified rules
    Args:
        matrix: Matrix to transform
        method: Transformation method (no, clip)
        clip_upper_value: clip upper value
    Returns:
        Transformed matrix
    """
    if method == 'no':
        return matrix
    elif method == 'clip':
        return matrix.clip(upper=clip_upper_value)
    elif method == 'log':
        return matrix.apply(np.log).clip(0, clip_upper_value)


def columns_to_lowercase(df):
    df.columns = [x.lower() for x in df.columns]
    return df


def apply_weights(df, weight='bm25'):
    """
    Function apply weights to user-item matrix
    Args:
        df: Matrix user-item
        weight: (bm25, tf-idf, normalize) - weight method
    Returns:
    Weighted user-item matrix
    """
    if weight == 'bm25':
        crd_list = list(df.index.values)
        plu_list = list(df.columns)
        matrix = pd.DataFrame(bm25_weight(sparse.csr_matrix(df.to_numpy(), dtype='float16'), B=0.9).toarray())
        matrix.columns = plu_list
        matrix.index = crd_list
        return matrix

    if weight == 'tf-idf':
        crd_list = list(df.index.values)
        plu_list = list(df.columns)
        matrix = pd.DataFrame(tfidf_weight(sparse.csr_matrix(df.to_numpy(), dtype='float16')).toarray())
        matrix.columns = plu_list
        matrix.index = crd_list
        return matrix

    if weight == 'normalize':
        crd_list = list(df.index.values)
        plu_list = list(df.columns)
        matrix = pd.DataFrame(normalize(sparse.csr_matrix(df.to_numpy(), dtype='float16')).toarray())
        matrix.columns = plu_list
        matrix.index = crd_list
        return matrix


def precision_at_k(preds, df_test, matrix, k=5, warm=True):
    """Calculates Precision@k, x%"""

    precision_list = []
    for user in matrix.index:
        if warm == True:
            if (user in df_test['user_id'].unique()):
                pred = preds[user].tolist()
                pred = pred[:k]  # @k
                true = df_test.loc[df_test['user_id'] == user, 'item_id'].values.tolist()

                guessed = [p in true for p in pred]
                precision = sum(guessed) / min(sum(true), k)
                precision_list.append(precision)
        else:
            pred = preds[user].tolist()
            pred = pred[:k]  # @k
            true = df_test.loc[df_test['user_id'] == user, 'item_id'].values.tolist()

            guessed = [p in true for p in pred]
            precision = sum(guessed) / sum(true)
            precision_list.append(precision)
    return np.round(np.mean(precision_list) * 100, 2)