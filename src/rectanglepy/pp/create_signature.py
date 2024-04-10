import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from pandas import DataFrame, Series
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score

from rectanglepy.tl.deconvolution import solve_qp

from .rectangle_signature import RectangleSignatureResult


def _convert_to_cpm(count_sc_data):
    return count_sc_data * 1e6 / np.sum(count_sc_data)


def _create_condition_number_matrix(de_adjusted, pseudo_signature: pd.DataFrame, max_gene_number: int) -> pd.DataFrame:
    genes = set()
    for annotation in de_adjusted:
        genes.update(_find_signature_genes(max_gene_number, de_adjusted[annotation]))
    sliced_sc_data = pseudo_signature.loc[list(genes)]
    return sliced_sc_data


def _create_condition_number_matrices(de_adjusted, pseudo_signature):
    condition_number_matrices = []
    de_adjusted_lengths = {annotation: len(de_adjusted[annotation]) for annotation in de_adjusted}
    longest_de_analysis = max(de_adjusted_lengths.values())

    loop_range = min(longest_de_analysis, 200)
    range_minimum = 30

    if loop_range < range_minimum:
        range_minimum = 8

    for i in range(range_minimum, loop_range):
        condition_number_matrices.append(_create_condition_number_matrix(de_adjusted, pseudo_signature, i))

    return condition_number_matrices, range_minimum


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


def _calculate_cluster_range(number_of_cell_types: int) -> tuple[int, int]:
    cluster_factor = 3
    if number_of_cell_types > 12:
        cluster_factor = 4
    if number_of_cell_types > 20:
        cluster_factor = 6
    if number_of_cell_types > 50:
        cluster_factor = 10
    min_number_clusters = max(
        3, number_of_cell_types - cluster_factor
    )  # we don't want to cluster too many cell types together
    max_number_clusters = number_of_cell_types - 1  # we want to have at least one cluster wih multiple cell types
    return min_number_clusters, max_number_clusters


def _create_fclusters(signature: pd.DataFrame, linkage_matrix) -> list[int]:
    min_number_clusters, max_number_clusters = _calculate_cluster_range(len(signature.columns))
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
    label_dict = dict(zip(signature.columns, labels))
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
    # if increase p-value and decrease log2FC until genes are found or the threshold is reached
    while len(adjusted_result) < 10 and (min_log2FC > 0.5 and max_p < 0.05):
        min_log2FC = max(min_log2FC - 0.1, 0.5)
        max_p = min(max_p + 0.001, 0.05)
        adjusted_result = de_analysis_result[
            (de_analysis_result["pvalue"] < max_p) & (de_analysis_result["log2_fc"] > min_log2FC)
        ]

    return adjusted_result


def _run_deseq2(countsig: pd.DataFrame, n_cpus: int = None) -> dict[str | int, pd.DataFrame]:
    results = {}
    count_df = countsig[countsig.sum(axis=1) > 0].T
    for i, cell_type in enumerate(countsig.columns):
        logger.info(f"Running DE analysis for {cell_type}")
        condition = np.zeros(len(countsig.columns))
        condition[i] = 1
        clinical_df = pd.DataFrame({"condition": condition}, index=countsig.columns)
        dds = DeseqDataSet(counts=count_df, metadata=clinical_df, design_factors="condition", quiet=True, n_cpus=n_cpus)
        dds.deseq2()
        stat_res = DeseqStats(dds, quiet=True)
        stat_res.summary(quiet=True)
        stat_res.lfc_shrink()
        results[cell_type] = stat_res.results_df

    return results


def _de_analysis(
    pseudo_count_sig, sc_data, annotations, p, lfc, optimize_cutoffs: bool, n_cpus: int = None, genes=None
) -> tuple[Series, dict[str, int] :, DataFrame | None]:
    logger.info("Starting DE analysis")
    deseq_results = _run_deseq2(pseudo_count_sig, n_cpus)
    optimization_results = None

    if optimize_cutoffs:
        logger.info("Optimizing cutoff parameters p and lfc")
        optimization_results = _optimize_parameters(sc_data, annotations, pseudo_count_sig, deseq_results, genes)
        p, lfc = optimization_results.iloc[0, 0:2]
        logger.info(f"Optimization done\n Best cutoffs  p: {p} and lfc: {lfc}")

    markers, marker_genes_per_cell_type = _get_marker_genes(deseq_results, lfc, p, pseudo_count_sig)
    return pd.Series(markers), marker_genes_per_cell_type, optimization_results


def _get_marker_genes(deseq_results, logfc, p, pseudo_count_sig):
    de_analysis_adjusted = {
        annotation: _filter_de_analysis_results(result, p, logfc) for annotation, result in deseq_results.items()
    }

    pseudo_cpm_sig = _convert_to_cpm(pseudo_count_sig)
    condition_number_matrices, range_minimum = _create_condition_number_matrices(de_analysis_adjusted, pseudo_cpm_sig)
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    optimal_condition_index = condition_numbers.index(min(condition_numbers))

    optimal_condition_number = optimal_condition_index + range_minimum
    logger.info(f"Optimal condition number: {optimal_condition_number}")
    optimal_condition_matrix = condition_number_matrices[optimal_condition_index]

    markers = optimal_condition_matrix.index
    marker_genes_per_cell_type = _count_marker_genes_per_cell_type(de_analysis_adjusted, optimal_condition_number)
    return markers, marker_genes_per_cell_type


def _count_marker_genes_per_cell_type(de_analysis_adjusted, optimal_condition_number: int) -> dict[str, int]:
    marker_genes_per_cell_type = {}
    for annotation, result in de_analysis_adjusted.items():
        marker_genes_per_cell_type[annotation] = min(len(result), optimal_condition_number)
    logger.info(f"Number of marker genes per cell type: {str(marker_genes_per_cell_type)}")
    return marker_genes_per_cell_type


def _create_bias_factors(countsig: pd.DataFrame, sc_counts: pd.DataFrame | np.ndarray, annotations) -> pd.Series:
    number_of_expressed_genes = (sc_counts > 0).sum(axis=0)
    number_of_expressed_genes = np.squeeze(np.asarray(number_of_expressed_genes))
    bias_factors = [
        np.mean(number_of_expressed_genes[[annotation == x for x in annotations]]) for annotation in countsig.columns
    ]
    mRNA_bias = bias_factors / np.min(bias_factors)
    return pd.Series(mRNA_bias, index=countsig.columns)


def _create_clustered_data(
    pseudo_sig_cpm: pd.DataFrame, marker_genes, annotations: pd.Series, sc_counts: pd.DataFrame | np.ndarray, genes
) -> (pd.DataFrame, pd.Series, list[int | str]):
    if len(pseudo_sig_cpm.columns) < 5:
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

    clustered_signature = _create_pseudo_count_sig(sc_counts, clustered_annotations, genes)

    return clustered_signature, clustered_annotations, assignments


def build_rectangle_signatures(
    adata: AnnData,
    cell_type_col: str = "cell_type",
    bulks: pd.DataFrame = None,
    *,
    optimize_cutoffs=True,
    layer: str = None,
    raw: bool = False,
    p=0.015,
    lfc=1.5,
    subsample: bool = False,
    sample_size: int = 1500,
    n_cpus: int = None,
) -> RectangleSignatureResult:
    r"""Builds rectangle signatures based on single-cell  count data and annotations.

    Parameters
    ----------
    adata
        The single-cell count data as a DataFrame. DataFrame must have the genes as index and cell identifier as columns. Each entry should be in raw counts.
    bulks
        The bulk data as a DataFrame. DataFrame must have the bulk identifier as index and the genes as columns. Each entry should be in transcripts per million (TPM).
    cell_type_col
        The annotations corresponding to the single-cell count data. Series data should have the cell identifier as index and the annotations as values.
    layer
        The Anndata layer to use for the single-cell data. Defaults to None.
    raw
        A flag indicating whether to use the raw Anndata data. Defaults to False.
    subsample : bool
        A flag indicating whether to balance the single-cell data. Defaults to False.
    sample_size : int
        The number of cells to balance the single-cell data to. Defaults to 1500. If cell number is less than this number it takes the original number of cells.
    optimize_cutoffs
        Indicates whether to optimize the p-value and log fold change cutoffs using gridsearch. Defaults to True.
    p
        The p-value threshold for the DE analysis (only used if optimize_cutoffs is False).
    lfc
        The log fold change threshold for the DE analysis (only used if optimize_cutoffs is False).
    n_cpus
        The number of cpus to use for the DE analysis. Defaults to the number of cpus available.

    Returns
    -------
    The result of the rectangle signature analysis which is of type RectangleSignatureResult.
    """
    if bulks is not None:
        genes = list(set(bulks.columns) & set(adata.var_names))
        genes = sorted(genes)
        assert len(genes) > 0, "No common genes between bulks and single-cell data"
        logger.info(f"Using {len(genes)} common genes between bulks and single-cell data")
        adata = adata[:, genes]

    annotations = adata.obs[cell_type_col]
    if subsample:
        annotations = _even(annotations, sample_size)
        adata = adata[annotations.index]

    if layer is not None:
        sc_counts = adata.layers[layer]
    elif raw:
        sc_counts = adata.raw.X
    else:
        sc_counts = adata.X

    sc_counts = sc_counts.T

    genes = adata.var_names

    pseudo_sig_counts = _create_pseudo_count_sig(sc_counts, annotations, genes)
    m_rna_biasfactors = _create_bias_factors(pseudo_sig_counts, sc_counts, annotations)

    marker_genes, marker_genes_per_cell_type, optimization_result = _de_analysis(
        pseudo_sig_counts, sc_counts, annotations, p, lfc, optimize_cutoffs, n_cpus, genes
    )
    pseudo_sig_cpm = _convert_to_cpm(pseudo_sig_counts)
    logger.info("Starting rectangle cluster analysis")

    clustered_signature, clustered_annotations, assignments = _create_clustered_data(
        pseudo_sig_cpm, marker_genes, annotations, sc_counts, genes
    )

    if len(clustered_signature) == 0:
        return RectangleSignatureResult(
            signature_genes=marker_genes,
            pseudobulk_sig_cpm=pseudo_sig_cpm,
            bias_factors=m_rna_biasfactors,
            marker_genes_per_cell_type=marker_genes_per_cell_type,
            optimization_result=optimization_result,
        )

    clustered_biasfact = _create_bias_factors(clustered_signature, sc_counts, clustered_annotations)
    clustered_genes = _de_analysis(clustered_signature, sc_counts, clustered_annotations, p, lfc, False)[0]
    clustered_signature = _convert_to_cpm(clustered_signature)
    return RectangleSignatureResult(
        signature_genes=marker_genes,
        pseudobulk_sig_cpm=pseudo_sig_cpm,
        bias_factors=m_rna_biasfactors,
        marker_genes_per_cell_type=marker_genes_per_cell_type,
        optimization_result=optimization_result,
        clustered_pseudobulk_sig_cpm=clustered_signature,
        clustered_signature_genes=clustered_genes,
        clustered_bias_factors=clustered_biasfact,
        cluster_assignments=assignments,
    )


def _create_pseudo_count_sig(sc_counts: np.ndarray, annotations: pd.Series, var_names) -> pd.DataFrame:
    unique_labels, label_indices = np.unique(annotations, return_inverse=True)
    grouped_sum = np.zeros((len(unique_labels), sc_counts.shape[0]))
    for i, _label in enumerate(unique_labels):
        label_columns = label_indices == i
        grouped_sum[i, :] = np.sum(sc_counts.T[label_columns, :], axis=0)
    grouped_sum = grouped_sum.T

    grouped_sum = pd.DataFrame(grouped_sum, index=var_names, columns=unique_labels).astype(int)
    return grouped_sum


def _optimize_parameters(
    sc_data: pd.DataFrame, annotations: pd.Series, pseudo_signature_counts: pd.DataFrame, de_results, genes=None
) -> pd.DataFrame:
    # if there are many cell types we relax the cutoffs
    lfcs = [x / 100 for x in range(140, 200, 10)]
    ps = [x / 1000 for x in range(15, 20, 1)]

    results = []
    logger.info("generating pseudo bulks")
    bulks, real_fractions = _generate_pseudo_bulks(sc_data, annotations, genes)
    for p in ps:
        for lfc in lfcs:
            rmse, pearson_r = _assess_parameter_fit(lfc, p, bulks, real_fractions, pseudo_signature_counts, de_results)
            logger.info(f"RMSE:{rmse}, Pearson R:{pearson_r} for p={p}, lfc={lfc}")
            results.append({"p": p, "lfc": lfc, "rmse": rmse, "pearson_r": pearson_r})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["pearson_r", "rmse"], ascending=[False, True])

    return results_df


def _assess_parameter_fit(
    lfc: float, p: float, bulks, real_fractions, pseudo_signature_counts, de_results
) -> (float, float):
    estimated_fractions = _generate_estimated_fractions(pseudo_signature_counts, bulks, p, lfc, de_results)

    real_fractions = real_fractions.sort_index()

    estimated_fractions = estimated_fractions.sort_index()

    rsme = np.sqrt(np.mean((real_fractions - estimated_fractions) ** 2))
    pearson_r = pearsonr(real_fractions.values.flatten(), estimated_fractions.values.flatten())[0]
    return rsme, pearson_r


def _generate_pseudo_bulks(sc_data, annotations, genes=None):
    number_of_bulks = 50
    split_size = 50
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
            random_annotations = np.random.choice(annotation_indices, number_of_cells, replace=False)
            random_annotations_indices = [annotations.index.get_loc(x) for x in random_annotations]
            indices.extend(random_annotations_indices)

        random_cells = sc_data[:, indices]  # Select columns by indices for ndarray
        random_cells_sum = random_cells.sum(axis=1)
        random_cells_sum = np.squeeze(np.asarray(random_cells_sum))

        pseudo_bulk = random_cells_sum * 1e6 / np.sum(random_cells_sum)
        bulks.append(pseudo_bulk)

        cell_fractions = np.array(cell_numbers) / np.sum(cell_numbers)
        cell_fractions = pd.Series(cell_fractions, index=annotations.unique())
        real_fractions.append(cell_fractions)
    bulks = pd.DataFrame(bulks).T
    if genes is not None:
        bulks.index = genes
    return bulks, pd.DataFrame(real_fractions).T


def _generate_estimated_fractions(pseudo_bulk_sig, bulks, p, logfc, de_results):
    marker_genes = pd.Series(_get_marker_genes(de_results, logfc, p, pseudo_bulk_sig)[0])
    signature = (pseudo_bulk_sig * 1e6 / np.sum(pseudo_bulk_sig)).loc[marker_genes]

    estimated_fractions = bulks.apply(lambda x: _solve_quadratic_programming(signature, x), axis=0)
    estimated_fractions.index = signature.columns

    return estimated_fractions


def _solve_quadratic_programming(signature, bulk):
    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    return solve_qp(signature, bulk)


def _reduce_to_common_genes(bulks: pd.DataFrame, sc_data: pd.DataFrame):
    genes = list(set(bulks.index) & set(sc_data.index))
    sc_data = sc_data.loc[genes].sort_index()
    bulks = bulks.loc[genes].sort_index()
    return bulks, sc_data


def _even(annotations: pd.Series, number: int) -> pd.Series:
    assert number > 0, "Number of cells must be greater than 0"
    annotation_counts = annotations.value_counts()
    selected_cells = []
    for annotation in annotation_counts.index:
        cells = annotations[annotations == annotation].index
        cells = np.random.choice(cells, min(number, len(cells)), replace=False)
        selected_cells.extend(cells)
    return annotations.loc[selected_cells]
