import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score

from rectanglepy.tl import correct_for_unknown_cell_content, solve_dampened_wsl

from .rectangle_signature import RectangleSignatureResult


def convert_to_cpm(count_sc_data):
    return count_sc_data * 1e6 / np.sum(count_sc_data)


def create_condition_number_matrix(
    de_adjusted, sc_data: pd.DataFrame, max_gene_number: int, annotations
) -> pd.DataFrame:
    # Use a set to automatically handle duplicates
    genes = set()
    for annotation in de_adjusted:
        genes.update(find_signature_genes(max_gene_number, de_adjusted[annotation]))
    sliced_sc_data = sc_data.loc[list(genes)]
    if any(pd.api.types.is_sparse(dtype) for dtype in sliced_sc_data.dtypes):
        sliced_sc_data = sliced_sc_data.sparse.to_dense()
    results = sliced_sc_data.groupby(annotations.values, axis=1).mean()
    return results


def create_condition_number_matrices(de_adjusted, sc_data, annotations):
    condition_number_matrices = []
    de_adjusted_lengths = {annotation: len(de_adjusted[annotation]) for annotation in de_adjusted}
    longest_de_analysis = max(de_adjusted_lengths.values())
    loop_range = min(longest_de_analysis, 200)
    min_number = 1 if loop_range < 50 else 50
    for i in range(min_number, loop_range):
        condition_number_matrices.append(create_condition_number_matrix(de_adjusted, sc_data, i, annotations))

    return condition_number_matrices


def find_signature_genes(number_of_genes, de_result):
    result_len = len(de_result)
    if result_len > 0:
        number_of_genes = min(number_of_genes, result_len)
        de_result = de_result.sort_values(by=["log2_fc"], ascending=False)
        return de_result["gene"][0:number_of_genes].values
    return []


def create_linkage_matrix(signature: pd.DataFrame):
    method = "complete"
    metric = "euclidean"
    return linkage((np.log(signature + 1)).T, method=method, metric=metric)


def sil_scores(X, Z, ts):
    scores = []
    for num_clust in ts:
        scores.append(silhouette_score(X, fcluster(Z, t=num_clust, criterion="maxclust")))
    return scores


def create_fclusters(signature, linkage_matrix) -> list[int]:
    assert len(signature.columns) > 5
    min_number_clusters = max(3, len(signature.columns) - 5)
    max_number_clusters = len(signature.columns) - 1
    scores = sil_scores((np.log(signature + 1)).T, linkage_matrix, range(min_number_clusters, max_number_clusters))
    # take max score
    cluster_number = scores.index(max(scores)) + min_number_clusters
    clusters = fcluster(linkage_matrix, criterion="maxclust", t=cluster_number)
    if len(set(clusters)) == 1:
        # default clustering clustered all cell types in same cluster, fallback to distance metric
        distance = linkage_matrix[0][2]
        clusters = fcluster(linkage_matrix, criterion="distance", t=distance)
    return list(clusters)


def get_fcluster_assignments(fclusters: list[int], signature_columns: pd.Index) -> list[int | str]:
    assignments = []
    clusters = list(fclusters)
    for cluster, cell in zip(fclusters, signature_columns):
        if clusters.count(cluster) > 1:
            assignments.append(cluster)
        else:
            assignments.append(cell)
    return assignments


def create_annotations_from_cluster_labels(labels, annotations, signature):
    assert len(labels) == len(signature.columns)
    label_dict = dict(zip(signature.columns, labels))
    assert set(annotations) == set(signature.columns)
    cluster_annotations = [str(label_dict[x]) for x in annotations]
    return pd.Series(cluster_annotations, index=annotations.index)


def pandas_to_r(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        return robjects.conversion.get_conversion().py2rpy(df)


def r_to_pandas(df_r):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        return robjects.conversion.get_conversion().rpy2py(df_r)


def generate_limma(countsig):
    countsig = countsig[countsig.sum(axis=1) > 0]
    edgeR = importr("edgeR")
    limma = importr("limma")

    countsig_r = pandas_to_r(countsig)
    dge_list = edgeR.DGEList(countsig_r)
    dge_list = edgeR.calcNormFactors(dge_list)

    cell_degs = {}
    for i, cell_type in enumerate(countsig.columns):
        groups = pd.Series(np.zeros(len(countsig.columns)))
        groups[i] = 1
        design = pd.DataFrame({"(intercept)": np.ones(len(countsig.columns)), "group" + str(i + 1): groups}).astype(
            "int"
        )

        v = limma.voom(dge_list, pandas_to_r(design), plot=False)
        fit = limma.lmFit(v, design=pandas_to_r(design))
        fit = limma.eBayes(fit)

        degs_r = limma.topTable(fit, coef=2, n=len(countsig))
        degs = r_to_pandas(degs_r)
        degs = degs[degs["logFC"] > 0]

        cell_degs[cell_type] = degs

    return cell_degs


def filter_de_analysis_results(de_analysis_result, p, logfc, annotation):
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

    print(f"Found {len(adjusted_result)} genes for {annotation}\n")
    return adjusted_result


def get_optimal_condition_number(condition_number_matrices, de_analysis_adjusted):
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    return condition_numbers.index(min(condition_numbers)) + 50


def get_limma_genes_condition(pseudo_count_sig, sc_data, annotations):
    limma_results = generate_limma(pseudo_count_sig)
    # save to pickle

    de_analysis_adjusted = {
        annotation: filter_de_analysis_results(result) for annotation, result in limma_results.items()
    }

    condition_number_matrices = create_condition_number_matrices(de_analysis_adjusted, sc_data, annotations)
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    optimal_condition_matrix = condition_number_matrices[condition_numbers.index(min(condition_numbers))]

    markers = optimal_condition_matrix.index
    return pd.Series(markers)


def generate_deseq2(countsig) -> dict[str | int, pd.DataFrame]:
    results = {}
    count_df = countsig[countsig.sum(axis=1) > 0].T
    for i, cell_type in enumerate(countsig.columns):
        condition = np.zeros(len(countsig.columns))
        condition[i] = 1
        clinical_df = pd.DataFrame({"condition": condition}, index=countsig.columns)
        dds = DeseqDataSet(counts=count_df, clinical=clinical_df, design_factors="condition")
        dds.deseq2()
        stat_res = DeseqStats(dds)
        stat_res.summary()
        stat_res.lfc_shrink()
        results[cell_type] = stat_res.results_df

    return results


def get_deseq2_genes_condition(pseudo_count_sig, sc_data, annotations, p, logfc, optimize_cutoffs: bool):
    deseq_results = generate_deseq2(pseudo_count_sig)
    if optimize_cutoffs:
        p, logfc = optimize_parameters(sc_data, annotations, pseudo_count_sig, deseq_results)
    markers = get_marker_genes(annotations, deseq_results, logfc, p, sc_data)
    return pd.Series(markers)


def get_marker_genes(annotations, deseq_results, logfc, p, sc_data):
    de_analysis_adjusted = {
        annotation: filter_de_analysis_results(result, p, logfc, annotation)
        for annotation, result in deseq_results.items()
    }
    condition_number_matrices = create_condition_number_matrices(de_analysis_adjusted, sc_data, annotations)
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    optimal_condition_index = condition_numbers.index(min(condition_numbers))
    optimal_condition_matrix = condition_number_matrices[optimal_condition_index]

    markers = optimal_condition_matrix.index
    return markers


def create_bias_factors(countsig):
    biasfactors = (countsig > 0).sum(axis=0)
    return biasfactors / biasfactors.min()


def build_rectangle_signatures(
    sc_counts: pd.DataFrame, annotations: pd.Series, p, lfc, optimize_cutoffs: bool
) -> RectangleSignatureResult:
    """Builds rectangle signatures based on single-cell count data and annotations.

    Parameters
    ----------
    sc_counts
        The single-cell count data as a DataFrame. DataFrame should have the genes as rows and cell as columns.
    annotations
        The annotations corresponding to the single-cell count data. Series should have the corresponding annotations for each cell.
    p
    p
        The p-value threshold for the DE analysis.
    lfc
        The log fold change threshold for the DE analysis.
    with_recursive_step
        Indicates whether to include the recursive clustering step. Defaults to True.
    optimize_cutoffs
        Indicates whether to optimize the p-value and log fold change cutoffs using gridsearch. Defaults to True.

    Returns
    -------
    The result of the rectangle signature analysis.
    """
    assert sc_counts is not None and annotations is not None

    pseudo_signature_counts = sc_counts.groupby(annotations.values, axis=1).sum()

    if any(pd.api.types.is_sparse(dtype) for dtype in pseudo_signature_counts.dtypes):
        # pseudo signature can be dense, this is also more straightforward for the R conversion
        pseudo_signature_counts = pseudo_signature_counts.sparse.to_dense()

    print("creating signature")
    biasfact = create_bias_factors(pseudo_signature_counts)
    genes = get_deseq2_genes_condition(pseudo_signature_counts, sc_counts, annotations, p, lfc, optimize_cutoffs)
    pseudo_signature_counts = convert_to_cpm(pseudo_signature_counts)

    if len(pseudo_signature_counts.columns) < 4:
        return RectangleSignatureResult(
            signature_genes=genes, pseudobulk_sig_cpm=pseudo_signature_counts, bias_factors=biasfact
        )

    print("creating clustered signature")
    linkage_matrix = create_linkage_matrix(pseudo_signature_counts.loc[genes])
    clusters = create_fclusters(pseudo_signature_counts.loc[genes], linkage_matrix)
    assignments, clustered_signature, clustered_biasfact, clustered_genes = None, None, None, None
    if len(set(clusters)) > 2:
        assignments = get_fcluster_assignments(clusters, pseudo_signature_counts.columns)
        clustered_annotations = create_annotations_from_cluster_labels(
            assignments, annotations, pseudo_signature_counts
        )

        clustered_signature = sc_counts.groupby(clustered_annotations.values, axis=1).sum()
        if any(pd.api.types.is_sparse(dtype) for dtype in clustered_signature.dtypes):
            # pseudo signature can be dense, this is also more straightforward for the R conversion
            clustered_signature = clustered_signature.sparse.to_dense()

        clustered_biasfact = create_bias_factors(clustered_signature)
        clustered_genes = get_deseq2_genes_condition(
            clustered_signature, sc_counts, clustered_annotations, p, lfc, False
        )
        clustered_signature = convert_to_cpm(clustered_signature)

    return RectangleSignatureResult(
        signature_genes=genes,
        pseudobulk_sig_cpm=pseudo_signature_counts,
        bias_factors=biasfact,
        clustered_pseudobulk_sig_cpm=clustered_signature,
        clustered_signature_genes=clustered_genes,
        clustered_bias_factors=clustered_biasfact,
        cluster_assignments=assignments,
    )


def optimize_parameters(
    sc_data: pd.DataFrame, annotations: pd.Series, pseudo_signature_counts: pd.DataFrame, de_results
) -> (float, float):
    lfcs = [x / 100 for x in range(80, 130, 10)]
    ps = [x / 1000 for x in range(18, 24, 1)]

    results = []
    for p in ps:
        for lfc in lfcs:
            print("Computing RMSE and correlation for p:", p, "and lfc:", lfc, "...")
            rmse, pearson_r = assess_parameter_fit(lfc, p, sc_data, annotations, pseudo_signature_counts, de_results)
            print("RMSE:", rmse, "Pearson R:", pearson_r)
            results.append({"p": p, "lfc": lfc, "rmse": rmse, "pearson_r": pearson_r})

    results_df = pd.DataFrame(results)

    best_r = results_df["pearson_r"].max()
    best_p = results_df[results_df["pearson_r"] == best_r]["p"].values[0]
    best_lfc = results_df[results_df["pearson_r"] == best_r]["lfc"].values[0]

    print("Best Pearson R was:", best_r, "with p:", best_p, "and lfc:", best_lfc)
    return best_p, best_lfc


def assess_parameter_fit(
    lfc: float, p: float, sc_data: pd.DataFrame, annotations: pd.Series, pseudo_signature_counts, de_results
) -> (float, float):
    bulks, real_fractions = generate_pseudo_bulks(sc_data, annotations)
    estimated_fractions = generate_estimated_fractions(
        pseudo_signature_counts, bulks, p, lfc, de_results, sc_data, annotations
    )

    real_fractions = real_fractions.sort_index()

    estimated_fractions = estimated_fractions.sort_index()

    rsme = calculate_rsme(real_fractions, estimated_fractions)
    pearson_r = calculate_correlation(real_fractions, estimated_fractions)
    return rsme, pearson_r


def generate_pseudo_bulks(sc_data, annotations):
    number_of_bulks = 30
    split_size = 60
    bulks = []
    real_fractions = []

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


def generate_signature(sc_data, annotations, de_results, pseudobulk_sig, logfc, p):
    marker_genes = pd.Series(get_marker_genes(annotations, de_results, logfc, p, sc_data))
    signature = (pseudobulk_sig * 1e6 / np.sum(pseudobulk_sig)).loc[marker_genes]
    return signature


def generate_estimated_fractions(pseudo_bulk_sig, bulks, p, logfc, de_results, sc_data, annotations):
    bias_factors = create_bias_factors(pseudo_bulk_sig)

    signature = generate_signature(sc_data, annotations, de_results, pseudo_bulk_sig, logfc, p) * bias_factors
    pseudo_sig_cpm = pseudo_bulk_sig * 1e6 / np.sum(pseudo_bulk_sig)

    estimated_fractions = bulks.apply(lambda x: solve_quadratic_programming(signature, x), axis=0)
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


def solve_quadratic_programming(signature, bulk):
    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    return solve_dampened_wsl(signature, bulk)


def calculate_rsme(real_fractions: pd.DataFrame, predicted_fractions: pd.DataFrame):
    return np.sqrt(np.mean((real_fractions - predicted_fractions) ** 2))


def calculate_correlation(real_fractions: pd.DataFrame, predicted_fractions: pd.DataFrame):
    return pearsonr(real_fractions.values.flatten(), predicted_fractions.values.flatten())[0]
