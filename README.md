[![tests](https://github.com/alonfnt/pbhc/actions/workflows/pytest.yml/badge.svg)](https://github.com/alonfnt/pbhc/actions/workflows/pytest.yml)
[![PyPI](https://img.shields.io/pypi/v/pbhc.svg)](https://pypi.org/project/pbhc/)

`pbhc` is a minimal parallel Bayesian hierarchical clustering (BHC) implementation in [JAX](https://github.com/jax-ml/jax) that runs on GPU/TPU/CPU.
Clustering is constrained to a graph: only clusters joined by an edge may merge, so on a pixel grid every cluster is spatially contiguous.
Instead of BHC's one-merge-at-a-time greedy loop, every cluster merges with its most probable neighbour in the same round (Borůvka-style), so the number of rounds grows roughly with the log of the number of points.

Each cluster follows a diagonal Normal-Normal model, `x ~ N(mu, sigma2)`, `mu ~ N(0, tau2)`, with a Dirichlet process prior of concentration `alpha` ([Heller & Ghahramani, 2005](https://doi.org/10.1145/1102351.1102389)).
Merges of several children at once use the product over children of [Bayesian rose trees](https://arxiv.org/abs/1203.3468), and merging stops when no merge has posterior probability above `threshold` (0.5 by default).

## Usage
```python
import jax
import jax.numpy as jnp
import pbhc

jax.config.update("jax_enable_x64", True)  # recommended for large inputs

# 100 points in a line: the first 50 around 0, the last 50 around 10
X = jax.random.normal(jax.random.key(0), (100, 3)).at[50:].add(10)

# edges between consecutive points (only connected clusters can merge)
i, j = jnp.arange(99), jnp.arange(1, 100)

# sigma2: spread inside a cluster, from differences between neighbours
# tau2: spread between clusters, from the variance of the whole data
sigma2 = 0.5 * jnp.mean((X[i] - X[j]) ** 2)
tau2 = jnp.var(X)

state = pbhc.fit(X, i, j, sigma2=sigma2, tau2=tau2)
print(state.labels)  # [0 0 ... 0 1 1 ... 1]
```

## Installation
`pbhc` can be installed from PyPI via `pip`
```
pip install pbhc
```

## Citation
If you use `pbhc` in your research and need to reference it, please cite it as follows:
```
@software{alonso_pbhc,
  author = {Alonso, Albert},
  title = {pbhc: Minimal Parallel Bayesian Hierarchical Clustering in JAX},
  url = {https://github.com/alonfnt/pbhc},
  version = {0.1.0}
}
```
