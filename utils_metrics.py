from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score as sk_f1
import numpy as np

def bio_prf1(y_true, y_pred):
    # y_* are lists of label sequences, e.g., [["O","B-ASP",...], ...]
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    return p, r, f

def cls_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    f1m = sk_f1(y_true, y_pred, average=average, zero_division=0)
    return acc, f1m
