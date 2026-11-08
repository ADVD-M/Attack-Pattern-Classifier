"""Fit one preprocessing object per training fold; never include metadata."""
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from load import FEATURES, CAT

LOG = 'duration src_bytes dst_bytes count srv_count dst_host_count dst_host_srv_count'.split()

class Features:
    def numeric(self, frame):
        data = frame[self.numeric_columns].astype(float).copy()
        for c in LOG:
            if c in data:
                data[c] = np.log1p(data[c])
        return data

    def fit(self, frame):
        self.fit_ids = frame.index.to_numpy().copy()
        self.dropped = [c for c in FEATURES if frame[c].nunique() == 1]
        self.numeric_columns = [c for c in FEATURES if c not in CAT and c not in self.dropped]
        self.categorical_columns = [c for c in CAT if c not in self.dropped]
        self.scaler = StandardScaler().fit(self.numeric(frame))
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32).fit(frame[self.categorical_columns])
        return self

    def transform(self, frame):
        return np.concatenate([self.scaler.transform(self.numeric(frame)).astype(np.float32), self.encoder.transform(frame[self.categorical_columns])], axis=1)
