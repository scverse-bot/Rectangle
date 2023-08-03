import math

import numpy as np
import pandas as pd
import quadprog
import statsmodels.api as sm
from loguru import logger

from .rectangle_signature import RectangleSignatureResult


def _scale_weights(weights):
    min_weight = np.nextafter(min(weights), np.float64(1.0))  # prevent division by zero
    return weights / min_weight


def solve_qp(signature, bulk, prev_assignments=None, prev_weights=None, gld=None, multiplier=None):
    """Performs quadratic programming optimization.

    Parameters
    ----------
    signature
        The signature data for deconvolution.
    bulk
        The bulk data for deconvolution.
    multiplier
        todo
    prev_weights
        todo
    prev_assignments
        todo
    gld
        todo


    Returns
    -------
    pd.Series: The corrected cell fractions.
    """
    # ------------------ QP-based deconvolution
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    if multiplier is None:
        a = np.dot(signature.T, bulk).astype("double")
        G = np.dot(signature.T, signature).astype("double")
    else:
        weights = np.square(1 / (signature @ gld))
        weights_dampened = np.clip(_scale_weights(weights), None, multiplier)
        W = np.diag(weights_dampened)
        G = np.dot(signature.T, np.dot(W, signature))
        a = np.dot(signature.T, np.dot(W, bulk))

    # Compute the constraints matrices
    n_genes = G.shape[0]
    # Constraint 1: sum of all fractions is below or equal to 1
    C1 = -np.ones((n_genes, 1))
    b1 = -np.ones(1)
    # Constraint 2: every fraction is greater or equal to 0
    C2 = np.eye(n_genes)
    b2 = np.zeros(n_genes)
    # Constraint 3: if a previous solution is provided, the new solution should be similar to the previous one
    prev_constraints, prev_b = [], []
    if prev_weights is not None:
        for cluster in prev_weights.index:
            C_upper = [-np.ones(1) if str(x) == str(cluster) else np.zeros(1) for x in prev_assignments]
            C_lower = [np.ones(1) if str(x) == str(cluster) else np.zeros(1) for x in prev_assignments]
            prev_constraints.extend([C_upper, C_lower])
            prev_weight = prev_weights.loc[cluster]
            # allow a 3% variation in the previous weights
            prev_weight_upper = min(1, prev_weight + 0.03)
            prev_weight_lower = max(0, prev_weight - 0.03)
            prev_b.extend([prev_weight_upper * np.ones(1) * -1, prev_weight_lower * np.ones(1)])

    C = np.concatenate((C1, C2, *prev_constraints), axis=1)
    b = np.concatenate((b1, b2, *prev_b), axis=0)

    scale = np.linalg.norm(G)
    solution = quadprog.solve_qp(G / scale, a / scale, C, b, factorized=False)
    return solution[0]


def _find_dampening_constant(signature, bulk, qp_gld):
    solutions_std = []
    np.random.seed(1)
    weights = np.square(1 / (np.dot(signature, qp_gld)))
    weights_scaled = _scale_weights(weights)
    weights_scaled_no_inf = weights_scaled[weights_scaled != np.inf]
    qp_gld_sum = sum(qp_gld)
    # try multiple values of the dampening constant (multiplier)
    # for each, calculate the variance of the dampened weighted solution for a subset of genes
    for i in range(math.ceil(np.log2(max(weights_scaled_no_inf)))):
        solutions = []
        multiplier = 2**i
        weights_dampened = np.array([multiplier if multiplier <= x else x for x in weights_scaled]).astype("double")
        for _ in range(86):
            subset = np.random.choice(len(signature), size=len(signature) // 2, replace=False)
            bulk_subset = bulk.iloc[list(subset)]
            signature_subset = signature.iloc[subset, :]
            fit = sm.WLS(bulk_subset, -1 + signature_subset, weights=weights_dampened[subset]).fit()
            solution = fit.params * qp_gld_sum / sum(fit.params)
            solutions.append(solution)
        solutions_df = pd.DataFrame(solutions)

        solutions_std.append(solutions_df.std(axis=0))
    solutions_std_df = pd.DataFrame(solutions_std)
    means = solutions_std_df.apply(lambda x: np.mean(x**2), axis=1)
    best_dampening_constant = means.idxmin()
    return best_dampening_constant


def calculate_dwls(signature, bulk, prev_assignments=None, prev_weights=None):
    """Calculate the deconvolution using dampened weighted least squares."""
    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    approximate_solution = solve_qp(signature, bulk, prev_assignments, prev_weights)
    dampening_constant = _find_dampening_constant(signature, bulk, approximate_solution)
    multiplier = 2**dampening_constant

    max_iterations = 1200
    convergence_threshold = 0.01
    change = 1
    iterations = 0
    while (change > convergence_threshold) and (iterations < max_iterations):
        dampened_solution = solve_qp(signature, bulk, prev_assignments, prev_weights, approximate_solution, multiplier)
        solution_averages = (dampened_solution + approximate_solution * 4) / 5
        change = np.linalg.norm(solution_averages - approximate_solution, 1)
        approximate_solution = solution_averages
        iterations += 1

    return pd.Series(approximate_solution, index=signature.columns)


def recursive_deconvolute(signatures: RectangleSignatureResult, bulk: pd.Series) -> pd.Series:
    """Performs recursive deconvolution using rectangle signatures and bulk data.

    Parameters
    ----------
    signatures
        The rectangle signature result containing the signature data and results.
    bulk
        The bulk data for deconvolution.

    Returns
    -------
        pd.Series: The estimated cell fractions resulting from deconvolution.

    Notes
    -----
        - The `signatures` parameter should be an instance of the `RectangleSignatureResult` class, representing the result of a rectangle signature analysis.
        - The `bulk` parameter should be a pandas Series representing the bulk data for deconvolution.
        - The function performs weighted dampened deconvolution using the original signature data, unless a clustered signature is available in the `signatures` object. In that case, recursive deconvolution is performed using both the original signature and clustered signature data.
        - The resulting estimated cell fractions are returned as a pandas Series.
        - If a pseudo signature is available in the `signatures` object, the estimated cell fractions are corrected for unknow cell content using the pseudo signature and the bulk data.
    """
    logger.info(f"Deconvolute fractions for bulk: {bulk.name}")

    pseudobulk_sig_cpm = signatures.pseudobulk_sig_cpm
    clustered_pseudobulk_sig_cpm = signatures.clustered_pseudobulk_sig_cpm

    signature = pseudobulk_sig_cpm.loc[signatures.signature_genes] * signatures.bias_factors
    start_fractions = calculate_dwls(signature, bulk)
    logger.info("Correct for unknown cell content")
    start_fractions = correct_for_unknown_cell_content(
        bulk, pseudobulk_sig_cpm, start_fractions, signatures.bias_factors
    )

    if clustered_pseudobulk_sig_cpm is None:
        logger.info("Direct deconvolution without recursive step returning result")
        return start_fractions

    logger.info("Recursive deconvolution")
    clustered_signature = (
        clustered_pseudobulk_sig_cpm.loc[signatures.clustered_signature_genes] * signatures.clustered_bias_factors
    )
    clustered_fractions = calculate_dwls(clustered_signature, bulk)
    recursive_fractions = calculate_dwls(signature, bulk, signatures.assignments, clustered_fractions)
    logger.info("Correct for unknown cell content")
    recursive_fractions = correct_for_unknown_cell_content(
        bulk, pseudobulk_sig_cpm, recursive_fractions, signatures.bias_factors
    )

    final_fractions = pd.concat([start_fractions, recursive_fractions]).groupby(level=0).mean()
    logger.info("Deconvolution complete")
    return final_fractions


def direct_deconvolute(signature: pd.DataFrame, bulk: pd.Series) -> pd.Series:
    """Performs direct deconvolution using a signature and bulk data.

    Parameters
    ----------
    signature
        The signature data for deconvolution.
    bulk
        The bulk data for deconvolution.

    Returns
    -------
    pd.Series: The estimated cell fractions resulting from deconvolution.

    Notes
    -----
        - The `signature` parameter should be a pandas DataFrame representing the signature data for deconvolution.
        - The `bulk` parameter should be a pandas Series representing the bulk data for deconvolution.
        - The function performs weighted dampened deconvolution using the provided signature and bulk data.
        - If a `pseudo_signature` is provided, the estimated cell fractions are corrected for unknown cell content using the pseudo signature and the bulk data.
        - The resulting estimated cell fractions are returned as a pandas Series.
    """
    assert (signature is not None) and (bulk is not None)

    logger.info("direct deconvolute")
    fractions = calculate_dwls(signature, bulk)

    return fractions


def correct_for_unknown_cell_content(
    bulk: pd.Series, pseudo_signature: pd.DataFrame, estimates: pd.Series, bias_factors: pd.Series
) -> pd.Series:
    """Performs direct deconvolution using a signature and bulk data.

    Parameters
    ----------
    bulk
        The bulk data for deconvolution.
    pseudo_signature
        The signature data for deconvolution.
    estimates
        The estimated cell fractions resulting from deconvolution.
    bias_factors
        The bias factors used for deconvolution.

    Returns
    -------
    pd.Series: The corrected cell fractions.
    """
    if estimates.sum() == 0:
        estimates_fix = estimates
        # analysis fails if all cell fractions are zero, so we set the unknown cell content to ß
        estimates_fix.loc["Unknown"] = 0
        return estimates_fix

    signature = pseudo_signature.sort_index()
    bulk = bulk.sort_index()

    # Reconstruct the bulk expression profiles through matrix multiplication
    # of the estimated cell fractions (weighted by the scaling factors) and
    # cell-type-specific expression profiles (i.e. signature)
    bulk_est = pd.Series(np.dot(signature, (estimates.T * bias_factors).T))
    bulk_est.index = signature.index

    # Calculate the unknown cellular content ad the difference of
    # per-sample overall expression levels in the true vs. reconstructed
    # bulk RNA-seq data, divided by the overall expression in the true bulk
    ukn_cc = (bulk - bulk_est).sum() / (bulk.sum())
    ukn_cc = max(0, ukn_cc)
    # Correct (i.e. scale) the cell fraction estimates so that their sum
    # equals 1 - the unknown cellular content estimated above
    estimates_fix = estimates / estimates.sum() * (1 - ukn_cc)
    estimates_fix.loc["Unknown"] = abs(ukn_cc)

    return estimates_fix
