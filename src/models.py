"""Fold-local selection and normal-only density estimation."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import Features

MODELS = [dict(id=i, display_name=n, kind=k) for i,n,k in [('rf','Random Forest','supervised'),('nb','Gaussian Naive Bayes','supervised'),('gmm','Gaussian Mixture','anomaly'),('iforest','Isolation Forest','anomaly'),('hybrid','RF + GMM','hybrid')]]

def ecdf(reference, values):
    return np.searchsorted(np.sort(reference), values, side='right') / len(reference)

def fit_score(frame, target, seed):
    prep = Features().fit(frame)
    x, z = prep.transform(frame), prep.transform(target)
    y = frame.y.to_numpy()
    normal = y == 0
    # Inner CV includes its own preprocessing, independent of outer held-out data.
    grid = [1e-9, 1e-7, 1e-5, 1e-3]
    values = np.zeros(len(grid))
    for a,b in StratifiedKFold(3, shuffle=True, random_state=seed).split(frame,y):
        inner = Features().fit(frame.iloc[a])
        xa, xb = inner.transform(frame.iloc[a]), inner.transform(frame.iloc[b])
        for j,v in enumerate(grid):
            nb = GaussianNB(var_smoothing=v).fit(xa,y[a])
            values[j] += roc_auc_score(y[b],nb.predict_proba(xb)[:,1])
    smoothing = grid[int(values.argmax())]
    rf = RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=seed).fit(x,y)
    nb = GaussianNB(var_smoothing=smoothing).fit(x,y)
    scores = dict(rf=rf.predict_proba(z)[:,1],nb=nb.predict_proba(z)[:,1])
    pca = PCA(svd_solver='full',random_state=seed).fit(x[normal])
    components = min(30, int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),.95)+1))
    xn = pca.transform(x[normal])[:,:components]
    zn = pca.transform(z)[:,:components]
    candidates = []
    for k in [1,2,4,8]:
        for reg in [1e-6,1e-4,1e-2]:
            try:
                g = GaussianMixture(n_components=k,covariance_type='full',reg_covar=reg,random_state=seed,max_iter=100).fit(xn)
                candidates.append((g.bic(xn),g))
                break
            except ValueError:
                if reg == 1e-2: raise
    _, gmm = min(candidates,key=lambda item:item[0])
    iso = IsolationForest(n_estimators=200,random_state=seed,n_jobs=-1).fit(x[normal])
    scores['gmm'] = ecdf(-gmm.score_samples(xn),-gmm.score_samples(zn))
    scores['iforest'] = ecdf(-iso.score_samples(x[normal]),-iso.score_samples(z))
    scores = {k:np.round(v,4) for k,v in scores.items()}
    scores['hybrid'] = np.maximum(scores['rf'],scores['gmm'])
    assert set(prep.fit_ids).isdisjoint(target.index) or target.attrs.get('external_test',False)
    audit = dict(train_rows=len(frame),normal_fit_rows=int(normal.sum()),anomaly_fit_ids=frame.index[normal].tolist(),preprocess_fit_ids=prep.fit_ids.tolist(),cdf_rows=int(normal.sum()),pca_components=components,pca_variance=float(pca.explained_variance_ratio_[:components].sum()),gmm_components=gmm.n_components,gmm_converged=bool(gmm.converged_),gmm_bic={str(g.n_components):float(bic) for bic,g in candidates},nb_smoothing=smoothing,features=x.shape[1],dropped=prep.dropped)
    return scores,audit
