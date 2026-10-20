from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


class PBHCState(NamedTuple):
    """Result of fit.

    Attributes:
        labels: Cluster of each point, 0..k-1.
        size: Size of the point's cluster.
        log_d: BHC's log d of the point's cluster.
        log_p: Log evidence log p(D | T) of the point's cluster.
        rounds: Number of merge rounds.
    """

    labels: jax.Array
    size: jax.Array
    log_d: jax.Array
    log_p: jax.Array
    rounds: jax.Array


def log_marginal(n: jax.Array, s: jax.Array, ss: jax.Array, sigma2, tau2) -> jax.Array:
    """
    Log p(D | H1) of each cluster, with x ~ N(mu, sigma2) and mu ~ N(0, tau2).

    Args:
        n: Points per cluster, shape (k,).
        s: Sum of the points, shape (k, n_features).
        ss: Sum of squares of the points, shape (k, n_features).
        sigma2: Variance of points around their cluster mean.
        tau2: Variance of the cluster means.

    Returns:
        Log marginal likelihood per cluster, shape (k,).
    """
    n = n[:, None]
    return jnp.sum(-0.5 * n * jnp.log(2 * jnp.pi * sigma2) - 0.5 * jnp.log1p(n * tau2 / sigma2)
                   - (ss - tau2 * s**2 / (sigma2 + n * tau2)) / (2 * sigma2), axis=1)


def _join(m, s, ss, log_d_children, log_p_children, sigma2, tau2, log_alpha, beta):
    """Merge children into one node. Returns log d, log p(D | T) and the merge log-odds."""
    log_d = jnp.logaddexp(log_alpha + gammaln(m), log_d_children)
    log_pi = log_alpha + gammaln(m) - log_d
    h1 = beta * log_marginal(m, s, ss, sigma2, tau2)
    rest = log_d_children - log_d + log_p_children  # log(1 - pi) = log(prod d_children / d), exact
    return log_d, jnp.logaddexp(log_pi + h1, rest), log_pi + h1 - rest


def _step(state, sigma2, tau2, log_alpha, beta, min_log_odds):
    """One round: each cluster merges with its best neighbour if r > threshold."""
    size, S, SS, log_d, log_p, ei, ej, label = state
    n, ids = len(size), jnp.arange(len(size))
    hyper = (sigma2, tau2, log_alpha, beta)
    _, _, lo = _join(size[ei] + size[ej], S[ei] + S[ej], SS[ei] + SS[ej],
                     log_d[ei] + log_d[ej], log_p[ei] + log_p[ej], *hyper)
    lo = jnp.where(ei != ej, lo, -jnp.inf)

    # each cluster points to its best neighbour if r > threshold; 2-cycles keep the smaller id as root
    ends, other = jnp.concatenate([ei, ej]), jnp.concatenate([ej, ei])
    lo2 = jnp.concatenate([lo, lo])
    best = jax.ops.segment_max(lo2, ends, n)
    target = jax.ops.segment_min(jnp.where(lo2 == best[ends], other, n), ends, n)
    target = jnp.where((best > min_log_odds) & (target < n), target, ids)
    target = jnp.where((target[target] == ids) & (ids < target), ids, target)
    root = jax.lax.fori_loop(0, n.bit_length(), lambda _, r: r[r], target)  # pointer jumping

    # merge every group that shares a root: sums of statistics, product over children in _join()
    seg = lambda v: jax.ops.segment_sum(v, root, n)  # noqa: E731
    kids = seg(jnp.ones_like(size))
    m, s, ss = seg(size), seg(S), seg(SS)
    new_d, new_p, _ = _join(m, s, ss, seg(log_d), seg(log_p), *hyper)
    grouped = kids > 1
    size, S, SS = jnp.where(grouped, m, size), jnp.where(grouped[:, None], s, S), jnp.where(grouped[:, None], ss, SS)
    log_d, log_p = jnp.where(grouped, new_d, log_d), jnp.where(grouped, new_p, log_p)
    ei, ej, label = root[ei], root[ej], root[label]
    return (size, S, SS, log_d, log_p, ei, ej, label), grouped.any()


@jax.jit
def fit(x: jax.Array, i: jax.Array, j: jax.Array, sigma2, tau2, alpha: float = 1.0,
        beta: float = 1.0, threshold: float = 0.5) -> PBHCState:
    """
    Bayesian hierarchical clustering of the rows of x, where only clusters joined by an edge can merge.

    Each round, every cluster merges with its best neighbour if the merge probability is above threshold.
    Enable float64 for large inputs.

    Args:
        x: Data, shape (n_samples, n_features).
        i: Edge sources, shape (n_edges,).
        j: Edge targets, shape (n_edges,).
        sigma2: Variance of points around their cluster mean.
        tau2: Variance of the cluster means.
        alpha: Dirichlet process concentration. Larger gives more clusters.
        beta: Likelihood temperature. Smaller gives larger clusters.
        threshold: Minimum merge probability.

    Returns:
        The fitted PBHCState.

    Each new input size compiles again. To reuse one compile across sizes, pad x with extra rows and the
    edges with self-loops (i == j), which never merge.
    """
    x = jnp.asarray(x, float)
    n = len(x)
    size = jnp.ones(n, x.dtype)
    log_d = jnp.full(n, jnp.log(alpha), x.dtype)
    log_p = beta * log_marginal(size, x, x**2, sigma2, tau2)
    state = (size, x, x**2, log_d, log_p, jnp.asarray(i, int), jnp.asarray(j, int), jnp.arange(n))
    hyper = (sigma2, tau2, jnp.log(alpha), beta, jnp.log(threshold) - jnp.log1p(-threshold))

    def body(carry):
        state, _, rounds = carry
        state, merged = _step(state, *hyper)
        return state, merged, rounds + 1

    state, _, rounds = jax.lax.while_loop(lambda c: c[1], body, (state, True, 0))

    root = state[7]
    is_root = jnp.zeros(n, int).at[root].set(1)
    labels = (jnp.cumsum(is_root) - 1)[root]
    return PBHCState(labels=labels, size=state[0][root], log_d=state[3][root], log_p=state[4][root],
                     rounds=rounds)

