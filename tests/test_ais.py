"""Unit tests - run with ``pytest -q`` (no dataset download needed)."""
import numpy as np
import pytest

from ais_patrec import (NOVEL, ClonalSelectionClassifier, ImmuneRecognizer,
                        NearestSelfDetector, VDetector)


@pytest.fixture
def ring():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, 400)
    r = 0.3 + rng.normal(0, 0.02, 400)
    return np.c_[0.5 + r * np.cos(t), 0.5 + r * np.sin(t)]


def blobs(rng, n=150):
    c = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]])
    X = np.vstack([rng.normal(ci, 0.05, (n, 2)) for ci in c])
    y = np.repeat(np.array(["a", "b", "c"], dtype=object), n)
    return X, y


@pytest.mark.parametrize("sampler", ["uniform", "self_proximal"])
def test_vdetector_tolerates_self(ring, sampler):
    vd = VDetector(self_radius=0.02, candidate_sampler=sampler, random_state=0).fit(ring)
    assert vd.n_detectors > 0
    # Negative selection guarantee: no detector covers a training self sample.
    assert vd.predict(ring).sum() == 0
    d = np.linalg.norm(vd.centers_[:, None] - ring[None], axis=2).min(1)
    assert np.all(vd.radii_ <= d - 0.02 + 1e-12)


def test_vdetector_flags_far_points(ring):
    vd = VDetector(self_radius=0.02, random_state=0).fit(ring)
    far = np.array([[0.5, 0.5], [0.02, 0.02], [0.98, 0.98]])
    assert vd.predict(far).tolist() == [1, 1, 1]
    assert vd.stop_reason_ in {"coverage_reached", "max_detectors", "max_candidates"}


def test_nearest_self(ring):
    ns = NearestSelfDetector(0.05).fit(ring)
    assert ns.predict(ring).sum() == 0
    assert ns.predict(np.array([[0.5, 0.5]]))[0] == 1


def test_clonal_learns_and_reports_affinity():
    rng = np.random.default_rng(1)
    X, y = blobs(rng)
    clf = ClonalSelectionClassifier(n_memory_per_class=8, n_generations=5, random_state=0).fit(X, y)
    assert clf.n_memory_cells == 24
    assert (clf.predict(X) == y).mean() > 0.97
    _, dist, rec = clf.predict_with_affinity(np.array([[0.5, 0.45], [5.0, 5.0]]))
    assert rec[1] == False and dist[1] > dist[0]  # noqa: E712 - far point not recognised


def test_maturation_moves_cells_toward_antigens():
    rng = np.random.default_rng(2)
    X, y = blobs(rng)
    a = ClonalSelectionClassifier(n_memory_per_class=5, n_generations=0, random_state=0).fit(X, y)
    b = ClonalSelectionClassifier(n_memory_per_class=5, n_generations=6, random_state=0).fit(X, y)
    def mean_nn(m):
        return np.linalg.norm(X[:, None] - m.memory_[None], axis=2).min(1).mean()
    assert mean_nn(b) < mean_nn(a)


def test_secondary_response_adds_class_without_forgetting():
    rng = np.random.default_rng(3)
    X, y = blobs(rng)
    keep = y != "c"
    clf = ClonalSelectionClassifier(n_memory_per_class=8, n_generations=4, random_state=0).fit(X[keep], y[keep])
    before = clf.memory_[clf.memory_labels_ == "a"].copy()
    clf.learn_new_pattern(X[y == "c"][:10], "c")
    assert "c" in set(clf.classes_)
    assert np.allclose(clf.memory_[clf.memory_labels_ == "a"], before)  # old memory frozen
    assert (clf.predict(X[y == "c"]) == "c").mean() > 0.9


def test_immune_recognizer_flags_novel():
    rng = np.random.default_rng(4)
    X, y = blobs(rng)
    y = np.where(y == "a", "normal", y).astype(object)
    rec = ImmuneRecognizer(vdetector=VDetector(self_radius=0.02, random_state=0),
                           clonal=ClonalSelectionClassifier(n_memory_per_class=8, n_generations=3,
                                                            recognition_quantile=0.95, random_state=0)).fit(X, y)
    out = rec.predict(np.array([[0.2, 0.2], [0.95, 0.95]]))
    assert out[0] == "normal"
    assert out[1] == NOVEL


def test_few_shot_class_is_recognised_not_novel():
    """Regression: a class learned from a few examples must get a non-zero
    recognition threshold, otherwise the immune recognizer calls it NOVEL."""
    rng = np.random.default_rng(5)
    X, y = blobs(rng)
    y = np.where(y == "a", "normal", y).astype(object)
    keep = y != "c"
    rec = ImmuneRecognizer(vdetector=VDetector(self_radius=0.02, random_state=0),
                           clonal=ClonalSelectionClassifier(n_memory_per_class=8, n_generations=3,
                                                            recognition_quantile=0.95, random_state=0))
    rec.vdetector.fit(X[y == "normal"]); rec.clonal.fit(X[keep], y[keep])
    shots = X[y == "c"][:5]
    rec.learn_new_pattern(shots, "c")
    assert rec.clonal.thresholds_["c"] > 0
    assert (rec.predict(X[y == "c"]) == "c").mean() > 0.5


def test_sqrt_allocation_gives_large_classes_more_memory():
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(0, 1, (900, 2)), rng.normal(5, 1, (100, 2)), rng.normal(-5, 1, (100, 2))])
    y = np.array(["big"] * 900 + ["s1"] * 100 + ["s2"] * 100, dtype=object)
    clf = ClonalSelectionClassifier(n_memory_per_class=10, memory_allocation="sqrt",
                                    n_generations=1, random_state=0).fit(X, y)
    counts = {c: int(np.sum(clf.memory_labels_ == c)) for c in clf.classes_}
    assert counts["big"] > counts["s1"] == counts["s2"]
