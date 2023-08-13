import numpy as np
import pandas as pd
from loguru import logger
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score

from .deconvolution import correct_for_unknown_cell_content, solve_qp
from .rectangle_signature import RectangleSignatureResult


def _convert_to_cpm(count_sc_data):
    return count_sc_data * 1e6 / np.sum(count_sc_data)


def _create_condition_number_matrix(
    de_adjusted, sc_data: pd.DataFrame, max_gene_number: int, annotations
) -> pd.DataFrame:
    genes = set()
    for annotation in de_adjusted:
        genes.update(_find_signature_genes(max_gene_number, de_adjusted[annotation]))
    sliced_sc_data = sc_data.loc[list(genes)]
    if any(pd.api.types.is_sparse(dtype) for dtype in sliced_sc_data.dtypes):
        sliced_sc_data = sliced_sc_data.sparse.to_dense()
    results = sliced_sc_data.groupby(annotations.values, axis=1).mean()
    return results


def _create_condition_number_matrices(de_adjusted, sc_data, annotations):
    condition_number_matrices = []
    de_adjusted_lengths = {annotation: len(de_adjusted[annotation]) for annotation in de_adjusted}
    longest_de_analysis = max(de_adjusted_lengths.values())
    loop_range = min(longest_de_analysis, 200)
    min_number = 3 if loop_range < 50 else 50
    for i in range(min_number, loop_range):
        condition_number_matrices.append(_create_condition_number_matrix(de_adjusted, sc_data, i, annotations))

    return condition_number_matrices


def _find_signature_genes(number_of_genes, de_result):
    result_len = len(de_result)
    if result_len > 0:
        number_of_genes = min(number_of_genes, result_len)
        de_result = de_result.sort_values(by=["log2_fc"], ascending=False)
        return de_result["gene"][0:number_of_genes].values
    return []


def _create_linkage_matrix(signature: pd.DataFrame):
    method = "complete"
    metric = "euclidean"
    return linkage((np.log(signature + 1)).T, method=method, metric=metric)


def _calculate_silhouette_scores(signature, linkage_matrix, cluster_range):
    scores = []
    for num_clust in cluster_range:
        scores.append(silhouette_score(signature, fcluster(linkage_matrix, t=num_clust, criterion="maxclust")))
    return scores


def _create_fclusters(signature: pd.DataFrame, linkage_matrix) -> list[int]:
    min_number_clusters = max(3, len(signature.columns) - 5)  # we don't want to cluster too many cell types together
    max_number_clusters = len(signature.columns) - 1  # we want to have at least one cluster wih multiple cell types
    cluster_range = range(min_number_clusters, max_number_clusters)

    scores = _calculate_silhouette_scores((np.log(signature + 1)).T, linkage_matrix, cluster_range)
    cluster_number = scores.index(max(scores)) + min_number_clusters
    clusters = fcluster(linkage_matrix, criterion="maxclust", t=cluster_number)

    if len(set(clusters)) == 1:
        # default clustering clustered all cell types in same cluster, fallback to distance metric
        distance = linkage_matrix[0][2]
        clusters = fcluster(linkage_matrix, criterion="distance", t=distance)

    return list(clusters)


def _get_fcluster_assignments(fclusters: list[int], signature_columns: pd.Index) -> list[int | str]:
    assignments = []
    clusters = list(fclusters)
    for cluster, cell in zip(fclusters, signature_columns):
        if clusters.count(cluster) > 1:
            assignments.append(cluster)
        else:
            assignments.append(cell)
    return assignments


def _create_annotations_from_cluster_labels(labels, annotations, signature):
    assert len(labels) == len(signature.columns)
    label_dict = dict(zip(signature.columns, labels))
    assert set(annotations) == set(signature.columns)
    cluster_annotations = [str(label_dict[x]) for x in annotations]
    return pd.Series(cluster_annotations, index=annotations.index)


def _filter_de_analysis_results(de_analysis_result, p, logfc):
    min_log2FC = logfc
    max_p = p
    de_analysis_result["log2_fc"] = de_analysis_result["log2FoldChange"]
    de_analysis_result["gene"] = de_analysis_result.index
    adjusted_result = de_analysis_result[
        (de_analysis_result["pvalue"] < max_p) & (de_analysis_result["log2_fc"] > min_log2FC)
    ]
    while len(adjusted_result) < 8 and (min_log2FC > 1 and max_p < 0.05):
        min_log2FC = max(min_log2FC - 0.1, 0.1)
        max_p = min(max_p + 0.001, 0.05)
        adjusted_result = de_analysis_result[
            (de_analysis_result["pvalue"] < max_p) & (de_analysis_result["log2_fc"] > min_log2FC)
        ]

    return adjusted_result


def _run_deseq2(countsig) -> dict[str | int, pd.DataFrame]:
    results = {}
    count_df = countsig[countsig.sum(axis=1) > 0].T
    for i, cell_type in enumerate(countsig.columns):
        logger.info(f"Running DE analysis for {cell_type}")
        condition = np.zeros(len(countsig.columns))
        condition[i] = 1
        clinical_df = pd.DataFrame({"condition": condition}, index=countsig.columns)
        dds = DeseqDataSet(counts=count_df, metadata=clinical_df, design_factors="condition", quiet=True)
        dds.deseq2()
        stat_res = DeseqStats(dds)
        stat_res.summary()
        stat_res.lfc_shrink()
        results[cell_type] = stat_res.results_df

    return results


def _de_analysis(pseudo_count_sig, sc_data, annotations, p, logfc, optimize_cutoffs: bool) -> pd.Series:
    logger.info("Starting DE analysis")
    deseq_results = _run_deseq2(pseudo_count_sig)
    logger.info("Finished DE analysis")

    if optimize_cutoffs:
        logger.info("Optimizing cutoff parameters p and lfc")
        p, logfc = _optimize_parameters(sc_data, annotations, pseudo_count_sig, deseq_results)
        logger.info(f"Optimization done\n Best cutoffs  p: {p} and lfc: {logfc}")
    markers = _get_marker_genes(annotations, deseq_results, logfc, p, sc_data)
    return pd.Series(markers)


def _get_marker_genes(annotations, deseq_results, logfc, p, sc_data):
    de_analysis_adjusted = {
        annotation: _filter_de_analysis_results(result, p, logfc) for annotation, result in deseq_results.items()
    }
    condition_number_matrices = _create_condition_number_matrices(de_analysis_adjusted, sc_data, annotations)
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    optimal_condition_index = condition_numbers.index(min(condition_numbers))
    optimal_condition_matrix = condition_number_matrices[optimal_condition_index]

    markers = optimal_condition_matrix.index
    return markers


def _create_bias_factors(countsig: pd.DataFrame) -> pd.Series:
    biasfactors = (countsig > 0).sum(axis=0)
    return biasfactors / biasfactors.min()


def _create_clustered_data(
    pseudo_sig_cpm: pd.DataFrame, marker_genes, annotations: pd.Series, sc_counts: pd.DataFrame
) -> (pd.DataFrame, pd.Series, list[int | str]):
    if len(pseudo_sig_cpm.columns) < 4:
        logger.info("Not enough cell types to perform clustering, returning direct rectangle signature")
        return pd.DataFrame(), pd.Series(), []

    linkage_matrix = _create_linkage_matrix(pseudo_sig_cpm.loc[marker_genes])

    clusters = _create_fclusters(pseudo_sig_cpm.loc[marker_genes], linkage_matrix)

    if len(set(clusters)) <= 2:
        logger.info("Not enough clusters to perform clustered signature analysis, returning direct rectangle signature")
        return pd.DataFrame(), pd.Series(), []

    logger.info("Starting clustered analysis")

    assignments = _get_fcluster_assignments(clusters, pseudo_sig_cpm.columns)
    clustered_annotations = _create_annotations_from_cluster_labels(assignments, annotations, pseudo_sig_cpm)
    clustered_signature = sc_counts.groupby(clustered_annotations.values, axis=1).sum()

    if any(pd.api.types.is_sparse(dtype) for dtype in clustered_signature.dtypes):
        # pseudo signature can be dense, this speeds up some calculations
        clustered_signature = clustered_signature.sparse.to_dense()
    return clustered_signature, clustered_annotations, assignments


def build_rectangle_signatures(
    sc_counts: pd.DataFrame, annotations: pd.Series, *, optimize_cutoffs=True, p=0.02, lfc=0.1
) -> RectangleSignatureResult:
    r"""Builds rectangle signatures based on single-cell  count data and annotations.

    Parameters
    ----------
    sc_counts
        The single-cell count data as a DataFrame. DataFrame must have the genes as index and cell identifier as columns. Each entry should be in raw counts.
    annotations
        The annotations corresponding to the single-cell count data. Series data should have the cell identifier as index and the annotations as values.
    optimize_cutoffs
        Indicates whether to optimize the p-value and log fold change cutoffs using gridsearch. Defaults to True.
    p
        The p-value threshold for the DE analysis (only used if optimize_cutoffs is False).
    lfc
        The log fold change threshold for the DE analysis (only used if optimize_cutoffs is False).

    Returns
    -------
    The result of the rectangle signature analysis which is of type RectangleSignatureResult.
    """
    assert sc_counts is not None and annotations is not None

    pseudo_sig_counts = sc_counts.groupby(annotations.values, axis=1).sum()

    if any(pd.api.types.is_sparse(dtype) for dtype in pseudo_sig_counts.dtypes):
        # pseudo signature can be dense, this speeds up the analysis
        pseudo_sig_counts = pseudo_sig_counts.sparse.to_dense()

    logger.info("Starting rectangle signature analysis")
    m_rna_biasfactors = _create_bias_factors(pseudo_sig_counts)
    marker_genes = _de_analysis(pseudo_sig_counts, sc_counts, annotations, p, lfc, optimize_cutoffs)
    pseudo_sig_cpm = _convert_to_cpm(pseudo_sig_counts)

    clustered_signature, clustered_annotations, assignments = _create_clustered_data(
        pseudo_sig_cpm, marker_genes, annotations, sc_counts
    )
    if len(clustered_signature) == 0:
        return RectangleSignatureResult(
            signature_genes=marker_genes, pseudobulk_sig_cpm=pseudo_sig_cpm, bias_factors=m_rna_biasfactors
        )

    clustered_biasfact = _create_bias_factors(clustered_signature)
    clustered_genes = _de_analysis(clustered_signature, sc_counts, clustered_annotations, p, lfc, False)

    return RectangleSignatureResult(
        signature_genes=marker_genes,
        pseudobulk_sig_cpm=pseudo_sig_cpm,
        bias_factors=m_rna_biasfactors,
        clustered_pseudobulk_sig_cpm=_convert_to_cpm(clustered_signature),
        clustered_signature_genes=clustered_genes,
        clustered_bias_factors=clustered_biasfact,
        cluster_assignments=assignments,
    )


def _optimize_parameters(
    sc_data: pd.DataFrame, annotations: pd.Series, pseudo_signature_counts: pd.DataFrame, de_results
) -> (float, float):
    """Optimizes the p-value and log fold change cutoffs for the DE analysis via gridsearch."""
    lfcs = [x / 100 for x in range(80, 130, 10)]
    ps = [x / 1000 for x in range(18, 24, 1)]

    results = []
    for p in ps:
        for lfc in lfcs:
            rmse, pearson_r = _assess_parameter_fit(lfc, p, sc_data, annotations, pseudo_signature_counts, de_results)
            logger.info(f"RMSE:{rmse}, Pearson R:{pearson_r} for p={p}, lfc={lfc}")
            results.append({"p": p, "lfc": lfc, "rmse": rmse, "pearson_r": pearson_r})

    results_df = pd.DataFrame(results)

    best_r = results_df["pearson_r"].max()
    best_p = results_df[results_df["pearson_r"] == best_r]["p"].values[0]
    best_lfc = results_df[results_df["pearson_r"] == best_r]["lfc"].values[0]

    return best_p, best_lfc


def _assess_parameter_fit(
    lfc: float, p: float, sc_data: pd.DataFrame, annotations: pd.Series, pseudo_signature_counts, de_results
) -> (float, float):
    bulks, real_fractions = _generate_pseudo_bulks(sc_data, annotations)
    estimated_fractions = _generate_estimated_fractions(
        pseudo_signature_counts, bulks, p, lfc, de_results, sc_data, annotations
    )

    real_fractions = real_fractions.sort_index()

    estimated_fractions = estimated_fractions.sort_index()

    rsme = _calculate_rsme(real_fractions, estimated_fractions)
    pearson_r = _calculate_correlation(real_fractions, estimated_fractions)
    return rsme, pearson_r


def _generate_pseudo_bulks(sc_data, annotations):
    number_of_bulks = 30
    split_size = 60
    bulks = []
    real_fractions = []
    np.random.seed(42)
    for _ in range(number_of_bulks):
        indices = []
        cell_numbers = []
        for annotation in annotations.unique():
            annotation_indices = annotations[annotations == annotation].index
            upper_limit = min(split_size, len(annotation_indices))
            number_of_cells = np.random.randint(0, upper_limit)
            cell_numbers.append(number_of_cells)
            random_annotation_indices = np.random.choice(annotation_indices, number_of_cells, replace=False)
            indices.extend(random_annotation_indices)

        random_cells = sc_data.loc[:, indices]
        random_cells_sum = random_cells.sum(axis=1)
        pseudo_bulk = random_cells_sum * 1e6 / np.sum(random_cells_sum)
        bulks.append(pseudo_bulk)

        cell_fractions = np.array(cell_numbers) / np.sum(cell_numbers)
        cell_fractions = pd.Series(cell_fractions, index=annotations.unique())
        real_fractions.append(cell_fractions)
    bulks = pd.DataFrame(bulks).T
    return bulks, pd.DataFrame(real_fractions).T


def _generate_estimated_fractions(pseudo_bulk_sig, bulks, p, logfc, de_results, sc_data, annotations):
    bias_factors = _create_bias_factors(pseudo_bulk_sig)

    marker_genes = pd.Series(_get_marker_genes(annotations, de_results, logfc, p, sc_data))
    signature = (pseudo_bulk_sig * 1e6 / np.sum(pseudo_bulk_sig)).loc[marker_genes]
    pseudo_sig_cpm = pseudo_bulk_sig * 1e6 / np.sum(pseudo_bulk_sig)

    estimated_fractions = bulks.apply(lambda x: _solve_quadratic_programming(signature, x), axis=0)
    estimated_fractions.index = signature.columns

    estimated_fractions_corrected = []
    for i in range(len(estimated_fractions.columns)):
        estimated_fractions_corrected.append(
            correct_for_unknown_cell_content(
                bulks.iloc[:, i], pseudo_sig_cpm, estimated_fractions.iloc[:, i], bias_factors
            )
        )

    estimated_fractions_corrected = pd.DataFrame(estimated_fractions_corrected).T
    estimated_fractions_corrected.drop("Unknown", axis=0, inplace=True)

    return estimated_fractions_corrected


def _solve_quadratic_programming(signature, bulk):
    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    return solve_qp(signature, bulk)


def _calculate_rsme(real_fractions: pd.DataFrame, predicted_fractions: pd.DataFrame):
    return np.sqrt(np.mean((real_fractions - predicted_fractions) ** 2))


def _calculate_correlation(real_fractions: pd.DataFrame, predicted_fractions: pd.DataFrame):
    return pearsonr(real_fractions.values.flatten(), predicted_fractions.values.flatten())[0]
