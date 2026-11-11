"""NSL-KDD loading and "shape-space" preprocessing.

Immunology analogy
------------------
Perelson & Oster's *shape-space* is the abstract space in which receptors and
antigens are compared.  Here a network connection record (41 features) is the
*antigen*; preprocessing maps it into a compact real-valued shape-space in
[0, 1]^k where Euclidean distance plays the role of (inverse) binding affinity.

Dataset: NSL-KDD (Tavallaee et al., 2009), a de-duplicated revision of KDD Cup
'99.  KDDTest+ contains attack *types* that never appear in KDDTrain+, which is
what lets us test "zero-day" behaviour honestly.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

NSL_KDD_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/{name}"
NSL_KDD_FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
# Heavy-tailed counters: log1p tames them before scaling.
HEAVY_TAILED = ["duration", "src_bytes", "dst_bytes", "count", "srv_count",
                "dst_host_count", "dst_host_srv_count", "hot",
                "num_compromised", "num_root", "num_file_creations",
                "num_access_files"]

# Standard 4-family taxonomy (Tavallaee et al., 2009). Types marked (test-only)
# never appear in KDDTrain+.
ATTACK_FAMILY = {
    "normal": "normal",
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probing / reconnaissance
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote-to-local
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L", "worm": "R2L",  # 'worm' is mapped inconsistently in the
                                    # literature (DoS or R2L); only 2 records.
    # User-to-root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}
FAMILIES = ["normal", "DoS", "Probe", "R2L", "U2R"]


def download_nsl_kdd(data_dir: str | os.PathLike = "data") -> dict[str, Path]:
    """Download KDDTrain+ / KDDTest+ if they are not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, name in NSL_KDD_FILES.items():
        path = data_dir / name
        if not path.exists():
            url = NSL_KDD_URL.format(name=name.replace("+", "%2B"))
            print(f"[data] downloading {url}")
            urllib.request.urlretrieve(url, path)
        paths[split] = path
    return paths


def load_nsl_kdd(data_dir: str | os.PathLike = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with an added ``family`` column."""
    paths = download_nsl_kdd(data_dir)
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(paths[split], header=None, names=COLUMNS)
        unknown = set(df["attack_type"]) - set(ATTACK_FAMILY)
        if unknown:
            raise ValueError(f"Unmapped attack types: {sorted(unknown)}")
        df["family"] = df["attack_type"].map(ATTACK_FAMILY)
        frames.append(df)
    train, test = frames
    train_types = set(train["attack_type"])
    test["novel_type"] = ~test["attack_type"].isin(train_types)
    train["novel_type"] = False
    return train, test


@dataclass
class ShapeSpace:
    """Maps raw connection records into a k-dimensional [0, 1] shape-space.

    Pipeline: log1p(heavy-tailed) -> one-hot(categoricals) -> standardise ->
    PCA(k) -> min-max to [0, 1].  The transform is fitted on *training* data
    only (optionally on self/normal only) so no test information leaks.

    Why PCA?  Negative selection degrades sharply in high dimensions
    (Stibor et al., 2005): random detectors either overlap self or leave
    "holes".  Projecting the ~120 one-hot dimensions onto the principal
    directions of variation keeps the geometry meaningful for hyperspheres.
    """

    n_components: int = 12
    standardize_categoricals: bool = False
    random_state: int = 0
    _onehot: OneHotEncoder = field(init=False, repr=False)
    _std: StandardScaler = field(init=False, repr=False)
    _pca: PCA = field(init=False, repr=False)
    _mm: MinMaxScaler = field(init=False, repr=False)

    def _numeric(self, df: pd.DataFrame) -> np.ndarray:
        num = df.drop(columns=CATEGORICAL + ["attack_type", "difficulty",
                                             "family", "novel_type"],
                      errors="ignore").copy()
        for col in HEAVY_TAILED:
            num[col] = np.log1p(num[col].astype(float))
        return num.to_numpy(dtype=float)

    def _design(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Numeric block standardised; one-hot block left as 0/1 unless
        ``standardize_categoricals`` (standardising rare one-hot columns
        inflates them into huge outliers and dominates PCA)."""
        num = self._numeric(df)
        if fit:
            self._onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat = self._onehot.fit_transform(df[CATEGORICAL])
        else:
            cat = self._onehot.transform(df[CATEGORICAL])
        X = np.hstack([num, cat])
        if fit:
            n_num = num.shape[1]
            self._std = StandardScaler().fit(X)
            if not self.standardize_categoricals:
                self._std.mean_[n_num:] = 0.0
                self._std.scale_[n_num:] = 1.0
        return self._std.transform(X)

    def fit(self, df: pd.DataFrame) -> "ShapeSpace":
        Z = self._design(df, fit=True)
        self._pca = PCA(n_components=self.n_components,
                        random_state=self.random_state).fit(Z)
        P = self._pca.transform(Z)
        self._mm = MinMaxScaler().fit(P)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        P = self._pca.transform(self._design(df))
        # Clip: unseen test traffic may fall slightly outside the unit cube.
        return np.clip(self._mm.transform(P), -0.5, 1.5)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def explained_variance(self) -> float:
        return float(self._pca.explained_variance_ratio_.sum())


def stratified_subsample(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Indices with at most ``per_class`` samples from each label."""
    idx = []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        take = min(per_class, members.size)
        idx.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(idx))
