"""Shared decision convention: flag when score >= threshold."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

RATIOS = [1, 2, 5, 10, 20, 50, 100]

def confusion(y, scores, threshold, r=10):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(scores) >= threshold
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, cost=(r * fn + fp) / len(y))

def metrics(y, scores, threshold, r=10):
    c = confusion(y, scores, threshold, r)
    tp, fp, fn, tn = (c[k] for k in ['tp', 'fp', 'fn', 'tn'])
    return dict(**c, accuracy=(tp+tn)/len(y), precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn), f1=2*tp/max(1,2*tp+fp+fn), fpr=fp/max(1,fp+tn), auc=float(roc_auc_score(y,scores)))

def optimal(y, scores, r):
    # Include every unique exported score and the all-negative decision.
    order = np.argsort(scores, kind='stable')
    s, y = np.asarray(scores)[order], np.asarray(y)[order]
    indices = np.r_[0, np.flatnonzero(np.diff(s)) + 1, len(s)]
    positives = np.r_[0, np.cumsum(y)]
    fn = positives[indices]
    fp = len(s) - indices - (positives[-1] - fn)
    thresholds = np.r_[s[indices[:-1]], 1.0001]
    return float(thresholds[np.argmin(r*fn+fp)])

def roc(y, scores):
    fpr, tpr, _ = roc_curve(y, scores)
    idx = np.unique(np.linspace(0, len(fpr)-1, min(200,len(fpr))).astype(int))
    return dict(fpr=fpr[idx].tolist(), tpr=tpr[idx].tolist())
