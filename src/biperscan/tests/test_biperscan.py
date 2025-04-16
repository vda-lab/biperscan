import pytest
import numpy as np
from functools import wraps
from tempfile import mkdtemp
from scipy.spatial.distance import pdist

from biperscan import BPSCAN, bpscan, lenses
from biperscan.api import KDTREE_VALID_METRICS, BALLTREE_VALID_METRICS


def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed.

    Parameters
    ----------
    func
    """

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib

            matplotlib.use("Agg")
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt

            plt.figure()
        except ImportError:
            pytest.skip("Matplotlib not available.")
        else:
            res = func(*args, **kwargs)
            plt.close("all")
            return res

    return run_test


def if_pandas(func):
    """Test decorator that skips test if pandas not installed."""

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import pandas
        except ImportError:
            pytest.skip("Pandas not available.")
        else:
            return func(*args, **kwargs)

    return run_test


def if_networkx(func):
    """Test decorator that skips test if networkx not installed."""

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import networkx
        except ImportError:
            pytest.skip("NetworkX not available.")
        else:
            return func(*args, **kwargs)

    return run_test


def if_pygraphviz(func):
    """Test decorator that skips test if networkx or pygraphviz is not installed."""

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import networkx
            import pygraphviz
        except ImportError:
            pytest.skip("NetworkX or pygraphviz not available.")
        else:
            return func(*args, **kwargs)

    return run_test


def make_branches(points_per_branch=30):
    # Control points for line segments that merge three clusters
    p0 = (0.13, -0.26)
    p1 = (0.24, -0.12)
    p2 = (0.32, 0.1)
    p3 = (0.13, 0.1)

    # Noisy points along lines between three clusters
    return np.concatenate(
        [
            np.column_stack(
                (
                    np.linspace(p_start[0], p_end[0], points_per_branch),
                    np.linspace(p_start[1], p_end[1], points_per_branch),
                )
            )
            + np.random.normal(size=(points_per_branch, 2), scale=0.01)
            for p_start, p_end in [(p0, p1), (p1, p2), (p1, p3)]
        ]
    )


np.random.seed(1)
X = np.concatenate(
    (
        make_branches(),
        make_branches()[:60] + np.array([0.3, 0]),
        np.column_stack(
            (np.random.uniform(0.1, 0.6, 10), np.random.uniform(-0.25, 0.1, 10))
        ),
    )
)
dists = pdist(X)


def test_base_params():
    c = BPSCAN(min_samples=5, min_cluster_size=20).fit(X)
    assert len(set(c.labels_)) == 10
    assert c.membership_.shape == (X.shape[0], 6)
    assert len(set(BPSCAN(min_samples=5, min_cluster_size=20).fit_predict(X))) == 10


def test_bad_args():
    with pytest.raises(ValueError):
        bpscan(X="fail")
    with pytest.raises(ValueError):
        bpscan(X=None)
    with pytest.raises(ValueError):
        bpscan(X, min_cluster_size=-1)
    with pytest.raises(ValueError):
        bpscan(X, min_cluster_size=0)
    with pytest.raises(ValueError):
        bpscan(X, min_cluster_size=1)
    with pytest.raises(ValueError):
        bpscan(X, min_cluster_size=2.0)
    with pytest.raises(ValueError):
        bpscan(X, min_cluster_size=X.shape[0] + 1)
    with pytest.raises(ValueError):
        bpscan(X, min_cluster_size="fail")
    with pytest.raises(ValueError):
        bpscan(X, min_samples=-1)
    with pytest.raises(ValueError):
        bpscan(X, min_samples=0)
    with pytest.raises(ValueError):
        bpscan(X, min_samples=1.0)
    with pytest.raises(ValueError):
        bpscan(X, min_samples="fail")
    with pytest.raises(ValueError):
        bpscan(X, distance_fraction=1)
    with pytest.raises(ValueError):
        bpscan(X, distance_fraction=-1.0)
    with pytest.raises(ValueError):
        bpscan(X, distance_fraction=-2.0)
    with pytest.raises(TypeError):
        bpscan(X, metric=None)
    with pytest.raises(ValueError):
        bpscan(X, metric="imperial")
    with pytest.raises(ValueError):
        bpscan(X, metric="minkowski", metric_kws=dict(p=-1))
    with pytest.raises(ValueError):
        bpscan(X, metric="minkowski", metric_kws=dict(p=-0.1))
    with pytest.raises(TypeError):
        bpscan(X, metric="minkowski", metric_kws=dict(p="fail"))
    with pytest.raises(TypeError):
        bpscan(X, metric="minkowski", metric_kws=dict(p=None))
    with pytest.raises(ValueError):
        BPSCAN(min_samples=5, min_cluster_size=20, metric="precomputed").fit(dists)


def test_missing_data():
    X_missing = X.copy()
    X_missing[1, :] = np.nan
    with pytest.raises(ValueError):
        BPSCAN(min_samples=5, min_cluster_size=20).fit(X_missing)


def test_precomputed_distances():
    c = BPSCAN(
        min_samples=5,
        min_cluster_size=20,
        metric="precomputed",
        lens="negative_eccentricity",
    ).fit(dists)
    assert len(set(c.labels_)) == 8
    assert c.membership_.shape == (X.shape[0], 6)


def test_custom_distances():
    def custom_metric(x, y):
        return np.abs(x - y).sum()

    l = BPSCAN(min_samples=5, min_cluster_size=20, metric=custom_metric).fit_predict(X)
    assert len(set(l)) > 1


def test_kdtree_distances():
    for m in KDTREE_VALID_METRICS:
        l = BPSCAN(min_samples=5, min_cluster_size=20, metric=m).fit_predict(X)
        assert len(set(l)) > 1


@pytest.mark.skip("Test too slow")
def test_balltree_distances():
    for m in BALLTREE_VALID_METRICS[::-1]:
        l = BPSCAN(min_samples=5, min_cluster_size=20, metric=m).fit_predict(X)
        assert len(set(l)) > 1


def test_precomputed_lens():
    lens = lenses.negative_distance_to_mean(X, None)
    l = BPSCAN(min_samples=5, min_cluster_size=20, lens=lens).fit_predict(X)
    assert len(set(l)) > 1


def test_custom_lens():
    l = BPSCAN(
        min_samples=5, min_cluster_size=20, lens=lenses.negative_distance_to_mean
    ).fit_predict(X)
    assert len(set(l)) > 1


def test_implemented_lenses():
    for l in lenses.available_lenses.keys():
        labels = BPSCAN(min_samples=5, min_cluster_size=20, lens=l).fit_predict(X)
        assert len(set(labels)) > 1


def test_memory():
    cachedir = mkdtemp()
    c1 = BPSCAN(memory=cachedir, min_samples=5, min_cluster_size=25).fit(X)
    c2 = BPSCAN(
        memory=cachedir, min_samples=5, min_cluster_size=25, distance_fraction=0.5
    ).fit(X)
    assert len(set(c1.labels_)) > 1
    assert len(set(c2.labels_)) > 1


def test_attribute_conversions():
    c = BPSCAN(min_samples=5, min_cluster_size=20).fit(X)
    mp = c.minimal_presentation_
    if_pandas(mp.as_pandas)()
    if_networkx(mp.as_networkx)()

    mh = c.merges_
    if_pandas(mh.as_pandas)()

    sh = c.simplified_merges_
    if_pandas(sh.as_pandas)()

    lh = c.linkage_hierarchy_
    if_pandas(lh.as_pandas)()
    if_networkx(lh.as_networkx)()


def test_minpres_plots():
    c = BPSCAN(min_samples=5, min_cluster_size=20).fit(X)
    mp = c.minimal_presentation_
    if_matplotlib(mp.plot_persistence_areas)(view_type="value")
    if_matplotlib(mp.plot_persistence_areas)(transposed=True)
    if_matplotlib(mp.plot_network)(
        layout={n: (x, y) for n, (x, y) in enumerate(zip(X[:, 0], X[:, 1]))}
    )
    if_pygraphviz(if_matplotlib(mp.plot_network))()


def test_merges_plots():
    c = BPSCAN(min_samples=5, min_cluster_size=20).fit(X)
    mh = c.merges_
    if_matplotlib(mh.plot_persistence_areas)(view_type="value")
    if_matplotlib(mh.plot_persistence_areas)(transposed=True)
    if_matplotlib(mh.plot_merges)(*X.T)


def test_simplified_plots():
    c = BPSCAN(min_samples=5, min_cluster_size=20).fit(X)
    sh = c.simplified_merges_
    if_matplotlib(sh.plot_persistence_areas)(view_type="value")
    if_matplotlib(sh.plot_persistence_areas)(transposed=True)
    if_matplotlib(sh.plot_merges)(*X.T)


def test_linkage_plots():
    c = BPSCAN(min_samples=5, min_cluster_size=20).fit(X)
    lh = c.linkage_hierarchy_
    if_matplotlib(lh.plot_persistence_areas)(view_type="value")
    if_matplotlib(lh.plot_persistence_areas)(transposed=True)
    if_pygraphviz(if_matplotlib(lh.plot_network))()


def test_unavailable_attributes():
    clusterer = BPSCAN()
    with pytest.raises(AttributeError):
        clusterer.minimal_presentation_
    with pytest.raises(AttributeError):
        clusterer.merge_hierarchy_
    with pytest.raises(AttributeError):
        clusterer.simplified_hierarchy_
    with pytest.raises(AttributeError):
        clusterer.linkage_hierarchy_
    with pytest.raises(AttributeError):
        clusterer.membership_
    with pytest.raises(AttributeError):
        clusterer.labels_at_depth(1)
    with pytest.raises(AttributeError):
        clusterer.first_nonzero_membership()
