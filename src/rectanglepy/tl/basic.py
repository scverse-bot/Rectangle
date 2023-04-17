import math

import numpy as np
import pandas as pd
import quadprog
import statsmodels.api as sm


def scale_weights(weights):
    min_weight = np.nextafter(min(weights), np.float64(1.0))  # prevent division by zero
    return weights / min_weight


def solve_dampened_wsl(signature, bulk, prev_assignments=None, prev_weights=None, gld=None, multiplier=None):
    # ------------------ QP-based deconvolution
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    if multiplier is None:
        a = np.dot(signature.T, bulk)
        G = np.dot(signature.T, signature)
    else:
        weights = np.square(1 / (signature @ gld))
        weights_dampened = np.clip(scale_weights(weights), None, multiplier)
        W = np.diag(weights_dampened)
        G = np.dot(signature.T, np.dot(W, signature))
        a = np.dot(signature.T, np.dot(W, bulk))

    # Compute the constraints matrices
    n_genes = G.shape[0]
    # Constraint 1: sum of all fractions is below or equal to 1
    C1 = -np.ones((n_genes, 1))
    b1 = -np.ones(1)
    # Constraint 2: every fraction is greater than 0
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


def find_dampening_constant(signature, bulk, qp_gld):
    np.random.seed(1)  # make deterministic

    weights = np.square(1 / (np.dot(signature, qp_gld)))
    weights_scaled = scale_weights(weights)
    weights_scaled_no_inf = weights_scaled[weights_scaled != np.inf]
    max_multiplier = min(math.ceil(np.log2(max(weights_scaled_no_inf))), 50)

    # try multiple values of the dampening constant (multiplier)
    # for each, calculate the variance of the dampened weighted solution for a subset of genes
    solutions_std = [
        calculate_dampening_constant(bulk, signature, sum(qp_gld), weights_scaled, 2**i)
        for i in range(max_multiplier)
    ]
    means = np.square(pd.DataFrame(solutions_std)).mean(axis=1)
    return means.idxmin()


def calculate_dampening_constant(bulk, signature, sum_qp_gld, weights_scaled, multiplier):
    weights_dampened = np.where(weights_scaled >= multiplier, multiplier, weights_scaled)
    solutions = [solve_wls(bulk, signature, sum_qp_gld, weights_dampened) for _ in range(60)]
    return np.std(pd.DataFrame(solutions))


def solve_wls(bulk, signature, sum_qp_gld, weights_dampened):
    subset = np.random.choice(len(signature), size=len(signature) // 2, replace=False)
    params = sm.WLS(bulk[subset], -1 + signature.iloc[subset, :], weights=weights_dampened[subset]).fit().params
    return params * sum_qp_gld / sum(params)


def weighted_dampened_deconvolute(signature, bulk, prev_assignments=None, prev_weights=None):
    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    approximate_solution = solve_dampened_wsl(signature, bulk, prev_assignments, prev_weights)
    dampening_constant = find_dampening_constant(signature, bulk, approximate_solution)
    multiplier = 2**dampening_constant

    max_iterations = 1000
    convergence_threshold = 0.01
    change = 1
    iterations = 0
    while (change > convergence_threshold) and (iterations < max_iterations):
        dampened_solution = solve_dampened_wsl(
            signature, bulk, prev_assignments, prev_weights, approximate_solution, multiplier
        )
        solution_averages = (dampened_solution + approximate_solution * 4) / 5
        change = np.linalg.norm(solution_averages - approximate_solution, 1)
        approximate_solution = solution_averages
        iterations += 1

    return pd.Series(approximate_solution, index=signature.columns)


def recursive_deconvolute(signatures, bulk, sc_data=None, annotations=None):
    assert (signatures is not None) and (bulk is not None)
    start_signature = signatures[0][0]

    print("deconvolute start fractions")
    start_fractions = weighted_dampened_deconvolute(start_signature, bulk)

    if len(signatures) == 1:
        print("Simple deconvolution without recursive step")
        return start_fractions

    print("deconvolute recursive step")
    clustered_signature = signatures[1][0]
    clustered_assignments = signatures[1][1]

    print("Recursive deconvolution")
    clustered_fractions = weighted_dampened_deconvolute(clustered_signature, bulk)
    recursive_fractions = weighted_dampened_deconvolute(
        start_signature, bulk, clustered_assignments, clustered_fractions
    )

    final_fractions = pd.concat([start_fractions, recursive_fractions]).groupby(level=0).mean()

    if sc_data is not None and annotations is not None:
        print("correct for unknown content")
        pseudo_signature = create_cpm_pseudo_signature(sc_data, annotations)
        final_fractions = correct_for_unknown_cell_content(bulk, pseudo_signature, final_fractions)

    return final_fractions


def direct_deconvolute(signature: pd.DataFrame, bulk: pd.Series, sc_data, annotations) -> pd.Series:
    assert (signature is not None) and (bulk is not None)

    print("direct deconvolute")
    fractions = weighted_dampened_deconvolute(signature, bulk)

    if sc_data is not None and annotations is not None:
        print("correct for unknown content")
        pseudo_signature = create_cpm_pseudo_signature(sc_data, annotations)
        fractions = correct_for_unknown_cell_content(bulk, pseudo_signature, fractions)

    return fractions


# Function to get a pseudobulk signature for each cell type
# starting from scRNA-seq data in count format.
# Unlike the signatures built by deconvolution methods,
# this one contains all genes in the input data.
# The final signature in output is in CPM format.
def create_cpm_pseudo_signature(sc_counts, annotations):
    sc_counts.columns = pd.MultiIndex.from_arrays([annotations, sc_counts.columns])
    df_mean = sc_counts.T.groupby(level=0).mean().T
    df_mean_cpm = df_mean.apply(lambda x: x / sum(x) * 1e6)
    return df_mean_cpm


def create_scaling_factors(pseudo_signature: pd.DataFrame) -> pd.Series:
    scaling = (pseudo_signature > 0).sum()
    return scaling / min(scaling)


def correct_for_unknown_cell_content(bulk, pseudo_signature, estimates):
    scaling_factors = create_scaling_factors(pseudo_signature)
    genes = list(set(pseudo_signature.index) & set(bulk.index))
    signature = pseudo_signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index()

    # Reconstruct the bulk expression profiles through matrix multiplication
    # of the estimated cell fractions (weighted by the scaling factors) and
    # cell-type-specific expression profiles (i.e. signature)
    bulk_est = pd.Series(np.dot(signature, (estimates.T * scaling_factors).T))
    bulk_est.index = signature.index

    # Calculate the unknown cellular content ad the difference of
    # per-sample overall expression levels in the true vs. reconstructed
    # bulk RNA-seq data, divided by the overall expression in the true bulk
    ukn_cc = (bulk - bulk_est).sum() / (bulk.sum())

    # Correct (i.e. scale) the cell fraction estimates so that their sum
    # equals 1 - the unknown cellular content estimated above
    estimates_fix = estimates / estimates.sum() * (1 - ukn_cc)
    estimates_fix.loc["Unknown"] = abs(ukn_cc)

    return estimates_fix
