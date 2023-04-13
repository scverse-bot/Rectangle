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
    return approximate_solution
