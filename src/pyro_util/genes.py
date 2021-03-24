from typing import Union

import numpy as np
import scipy.sparse
import scipy.stats


def poisson_fit(umis: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    """Takes an array of UMI counts and calculates per-gene deviation from a poisson
    distribution representing even expression across all cells.

    :param umis: Unscaled UMI counts for ``n_cells * n_genes``
    :return: An array of logp-values of size ``n_genes``
    """
    if not scipy.sparse.issparse(umis):
        kwargs = {"keepdims": True}
    else:
        kwargs = {}

    n_cells = umis.shape[0]
    pct = np.asarray(((umis > 0).sum(0) / n_cells)).flatten()
    exp = umis.sum(0, **kwargs) / umis.sum()
    numis = umis.sum(1, **kwargs)

    prob_zero = np.asarray(np.exp(-np.dot(exp.T, numis.T)))
    exp_pct_nz = np.asarray((1 - prob_zero).mean(1)).flatten()

    var_pct_nz = np.asarray((prob_zero * (1 - prob_zero)).mean(1)).flatten() / n_cells
    std_pct_nz = np.sqrt(var_pct_nz)

    exp_p = np.ones_like(pct)
    ix = np.asarray(std_pct_nz != 0).flatten()
    exp_p[ix] = scipy.stats.norm.logcdf(
        pct[ix], loc=exp_pct_nz[ix], scale=std_pct_nz[ix]
    )

    return exp_p


def binomial_deviance(umis: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    """Takes an array of UMI counts and calculates per-gene deviance from expected value
    if genes were constantly expressed. Based on code from F Will Townes at
        https://github.com/willtownes/glmpca

    Note that the scale of the resulting deviance values will depend on the data size.

    :param umis: Unscaled UMI counts for ``n_cells * n_genes``
    :return: An array of deviances of size ``n_genes``
    """

    if not scipy.sparse.issparse(umis):
        n_i = umis.sum(1, keepdims=True)  # counts per cell
        P = umis / n_i

        L1P = np.log1p(-P)

        ll_sat = np.sum(umis * (np.log(np.where(P, P, 1)) - L1P) + n_i * L1P, axis=0)
    else:
        n_i = umis.sum(1)
        P = umis.multiply(1 / n_i).tocsr()
        L1P = (-P).log1p().tocsr()

        ll_sat = umis.copy()
        ll_sat.data *= np.log(P.data) - L1P.data
        ll_sat = np.asarray((ll_sat + L1P.multiply(n_i)).sum(axis=0)).squeeze()

    feature_sums = np.asarray(umis.sum(0)).squeeze()

    n = n_i.sum()

    pi_j = feature_sums / n
    ix = pi_j > 0.0

    l1p = np.log(1 - pi_j)

    ll_null = np.zeros_like(feature_sums)
    ll_null[ix] = feature_sums[ix] * (np.log(pi_j[ix]) - l1p[ix]) + n * l1p[ix]

    return 2 * (ll_sat - ll_null)
