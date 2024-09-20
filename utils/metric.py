import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import f1_score


def HR(truth, pred, Ks):
    hit_K = {}
    for K in Ks:
        top_K_pred = pred[:, :K]
        hit = 0
        for i, pred_i in enumerate(top_K_pred):
            if truth[i] in pred_i:
                hit += 1
        hit_K[K] = hit / pred.shape[0]
    return hit_K


def Cls_HR(truth, pred):
    hit = 0
    for i, pred_i in enumerate(pred):
        if truth[i] in pred_i:
            hit += 1
    return hit / len(truth)


def MAPE(truth, pred):
    return mean_absolute_percentage_error(truth, pred)


def MAE(truth, pred):
    return mean_absolute_error(truth, pred)


def RMSE(truth, pred):
    return math.sqrt(mean_squared_error(truth, pred))


def F1(truth, pred, num_user):
    return f1_score(truth, pred, average='micro', labels=np.arange(num_user).tolist()), \
        f1_score(truth, pred, average='macro', labels=np.arange(num_user).tolist())
