import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pbhc import fit, log_marginal  # noqa: E402

KEY = jax.random.key(42)


def path(n):
    return jnp.arange(n - 1), jnp.arange(1, n)


def blobs(n, d=3, gap=10.0):
    x = jax.random.normal(KEY, (2 * n, d))
    return x.at[n:].add(gap)


def test_log_marginal_matches_scipy():
    from scipy.stats import multivariate_normal
    sigma2, tau2, n = 0.5, 2.0, 7
    x = jax.random.normal(KEY, (n, 3))
    lm = log_marginal(jnp.array([n]), x.sum(0)[None], (x**2).sum(0)[None], sigma2, tau2)[0]
    cov = sigma2 * np.eye(n) + tau2 * np.ones((n, n))
    ref = sum(multivariate_normal(np.zeros(n), cov).logpdf(np.asarray(x[:, k])) for k in range(3))
    assert jnp.allclose(lm, ref, rtol=1e-10)


def test_two_blobs_on_path():
    x = blobs(50)
    state = fit(x, *path(100), sigma2=1.0, tau2=100.0)
    labels = np.asarray(state.labels)
    assert len(np.unique(labels)) == 2
    assert np.all(labels[:50] == labels[0]) and np.all(labels[50:] == labels[50])
    assert np.all(np.asarray(state.size) == 50)


def test_no_edges_no_merges():
    x = jax.random.normal(KEY, (20, 2))
    state = fit(x, jnp.zeros(0, int), jnp.zeros(0, int), sigma2=1.0, tau2=1.0)
    assert len(np.unique(state.labels)) == 20
    assert state.rounds == 1


def test_disconnected_components_never_merge():
    x = jnp.zeros((20, 2))
    i, j = path(10)
    state = fit(x, jnp.concatenate([i, i + 10]), jnp.concatenate([j, j + 10]), sigma2=1.0, tau2=1.0)
    labels = np.asarray(state.labels)
    assert len(np.unique(labels)) == 2
    assert labels[0] != labels[10]


@pytest.mark.parametrize("n", [5, 17, 100, 333])
def test_labels_contiguous(n):
    x = jax.random.normal(KEY, (n, 4)) * 3
    state = fit(x, *path(n), sigma2=1.0, tau2=4.0)
    labels = np.asarray(state.labels)
    assert labels.shape == (n,)
    assert set(np.unique(labels)) == set(range(labels.max() + 1))


def test_edge_order_invariant():
    x = blobs(30)
    i, j = path(60)
    a = fit(x, i, j, sigma2=1.0, tau2=100.0).labels
    b = fit(x, j[::-1], i[::-1], sigma2=1.0, tau2=100.0).labels
    assert jnp.array_equal(a, b)


def test_alpha_controls_number_of_clusters():
    x = jax.random.normal(KEY, (200, 2)) * 2
    k = [len(np.unique(fit(x, *path(200), sigma2=1.0, tau2=4.0, alpha=a).labels)) for a in (1e-3, 1e3)]
    assert k[0] <= k[1]


def test_threshold_controls_number_of_clusters():
    x = jax.random.normal(KEY, (200, 2)) * 2
    k = [len(np.unique(fit(x, *path(200), sigma2=1.0, tau2=4.0, threshold=t).labels)) for t in (0.01, 0.99)]
    assert k[0] <= k[1]


def test_jit():
    x = blobs(20)
    a = fit(x, *path(40), sigma2=1.0, tau2=100.0)
    b = jax.jit(fit)(x, *path(40), sigma2=1.0, tau2=100.0, alpha=1.0)
    assert jnp.array_equal(a.labels, b.labels)
