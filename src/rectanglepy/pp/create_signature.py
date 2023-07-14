import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

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
    max_gene_number = 50
    condition_number_matrices = []
    de_adjusted_lengths = {annotation: len(de_adjusted[annotation]) for annotation in de_adjusted}
    longest_de_analysis = max(de_adjusted_lengths.values())
    loop_range = max_gene_number + 1 if longest_de_analysis < max_gene_number else min(longest_de_analysis + 1, 200)

    for i in range(max_gene_number, loop_range):
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
    print(f"Found {len(adjusted_result)} genes for {annotation}\n")
    return adjusted_result


def get_optimal_condition_number(condition_number_matrices, de_analysis_adjusted):
    """Helper function to calculate the optimal condition number."""
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    smallest_de_analysis = min([len(de_analysis_adjusted[annotation]) for annotation in de_analysis_adjusted])
    return condition_numbers.index(min(condition_numbers)) + 1 + min([49, smallest_de_analysis - 1])


def get_limma_genes_condition(pseudo_count_sig, sc_data, annotations):
    limma_results = generate_limma(pseudo_count_sig)
    de_analysis_adjusted = {
        annotation: filter_de_analysis_results(result) for annotation, result in limma_results.items()
    }

    condition_number_matrices = create_condition_number_matrices(de_analysis_adjusted, sc_data, annotations)
    optimal_condition_number = get_optimal_condition_number(condition_number_matrices, de_analysis_adjusted)

    markers = create_condition_number_matrix(de_analysis_adjusted, sc_data, optimal_condition_number, annotations).index
    return pd.Series(markers)


def generate_deseq2(countsig):
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


def get_deseq2_genes_condition(pseudo_count_sig, sc_data, annotations, p, logfc):
    deseq_results = generate_deseq2(pseudo_count_sig)
    # write_limma_results
    import pickle

    with open("./deseq.pickle", "wb") as handle:
        pickle.dump(deseq_results, handle)
    de_analysis_adjusted = {
        annotation: filter_de_analysis_results(result, p, logfc, annotation)
        for annotation, result in deseq_results.items()
    }

    condition_number_matrices = create_condition_number_matrices(de_analysis_adjusted, sc_data, annotations)
    optimal_condition_number = get_optimal_condition_number(condition_number_matrices, de_analysis_adjusted)

    markers = create_condition_number_matrix(de_analysis_adjusted, sc_data, optimal_condition_number, annotations).index
    return pd.Series(markers)


def create_bias_factors(countsig):
    biasfactors = (countsig > 0).sum(axis=0)
    return biasfactors / biasfactors.min()


def build_rectangle_signatures(
    sc_counts: pd.DataFrame, annotations: pd.Series, p, logfc, with_recursive_step=True
) -> RectangleSignatureResult:
    """Builds rectangle signatures based on single-cell count data and annotations.

    Parameters
    ----------
    sc_counts
        The single-cell count data as a DataFrame. DataFrame should have the genes as rows and cell as columns.
    annotations
        The annotations corresponding to the single-cell count data. Series should have the corresponding annotations for each cell.
    p
        The p-value threshold for the DE analysis.
    logfc
        The log fold change threshold for the DE analysis.
    with_recursive_step
        Indicates whether to include the recursive clustering step. Defaults to True.

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
    genes = get_deseq2_genes_condition(pseudo_signature_counts, sc_counts, annotations, p, logfc)
    pseudo_signature_counts = convert_to_cpm(pseudo_signature_counts)

    if not with_recursive_step or len(pseudo_signature_counts.columns) < 4:
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
        clustered_genes = get_deseq2_genes_condition(clustered_signature, sc_counts, clustered_annotations, p, logfc)
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
