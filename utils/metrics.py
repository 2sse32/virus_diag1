import numpy as np


def accuracy(y, pred):
    return float((y == pred).mean())


def precision(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    return float(tp / max(pred.sum(), 1))


def recall(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    return float(tp / max(y.sum(), 1))


def f1(y, pred):
    p, r = precision(y, pred), recall(y, pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0
