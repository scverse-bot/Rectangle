import numpy as np
import pandas as pd
import rpy2.robjects.packages as rpackages
from anndata import AnnData
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

from .rectangle_signature import RectangleSignatureResult


def convert_to_cpm(count_sc_data):
    return count_sc_data * 1e6 / np.sum(count_sc_data)


def check_mast_install():
    if not rpackages.isinstalled("BiocManager"):
        print("BiocManager not installed")
    if not rpackages.isinstalled("edgeR"):
        print("edgeR is not installed")


def create_condition_number_matrix(
    de_adjusted, sc_data: pd.DataFrame, max_gene_number: int, annotations
) -> pd.DataFrame:
    genes = [find_signature_genes(max_gene_number, de_adjusted[annotation]) for annotation in de_adjusted]
    genes = list({gene for sublist in genes for gene in sublist})

    signature_columns = {
        annotation: sc_data.loc[genes, annotations == annotation].mean(axis=1) for annotation in de_adjusted
    }

    return pd.DataFrame(signature_columns)


def create_condition_number_matrices(de_adjusted, sc_data, annotations):
    max_gene_number = 50
    condition_number_matrices = []
    longest_de_analysis = max([len(de_adjusted[annotation]) for annotation in de_adjusted])
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


def build_rectangle_signatures_adata(
    adata: AnnData, convert_to_cpm=True, with_recursive_step=True
) -> RectangleSignatureResult:
    return build_rectangle_signatures(adata.to_df().T, adata.obs.iloc[:, 0], convert_to_cpm, with_recursive_step)


def generate_limma(countsig):
    countsig = countsig[countsig.sum(axis=1) > 0]
    edgeR = importr("edgeR")
    importr("base")
    limma = importr("limma")
    importr("stats")
    with localconverter(robjects.default_converter + pandas2ri.converter):
        countsig_r = robjects.conversion.get_conversion().py2rpy(countsig)
    dge_list = edgeR.DGEList(countsig_r)
    dge_list = edgeR.calcNormFactors(dge_list)
    cell_degs = {}
    for i in range(0, len(countsig.columns)):
        cell_type = countsig.columns[i]
        groups = pd.Series(np.zeros(len(countsig.columns)))
        groups[i] = 1
        ones = pd.Series(np.ones(len(countsig.columns)))
        design = pd.DataFrame({"(intercept)": ones, "group" + str(i + 1): groups}).astype("int")
        with localconverter(robjects.default_converter + pandas2ri.converter):
            design_r = robjects.conversion.get_conversion().py2rpy(design)

        v = limma.voom(dge_list, design_r, plot=False)
        fit = limma.lmFit(v, design=design_r)
        fit = limma.eBayes(fit)
        degs_r = limma.topTable(fit, coef=2, n=len(countsig))
        # convert back to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            degs = robjects.conversion.get_conversion().rpy2py(degs_r)

        degs = degs[degs["logFC"] > 0]

        cell_degs[cell_type] = degs
    return cell_degs


def get_limma_genes_condition(pseudo_count_sig, sc_data, annotations):
    limma = generate_limma(pseudo_count_sig)

    min_log2FC = 1
    max_p = 0.02

    de_analysis_adjusted = {}
    for annotation in limma.keys():
        de_analysis_result = limma[annotation]
        de_analysis_result["log2_fc"] = de_analysis_result["logFC"]
        de_analysis_result["gene"] = list(de_analysis_result.index)
        de_analysis_adjusted[annotation] = de_analysis_result[
            (de_analysis_result["P.Value"] < max_p) & (de_analysis_result["logFC"] > min_log2FC)
        ]

    condition_number_matrices = create_condition_number_matrices(de_analysis_adjusted, sc_data, annotations)
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    smallest_de_analysis = min([len(de_analysis_adjusted[annotation]) for annotation in de_analysis_adjusted])
    optimal_condition_number = condition_numbers.index(min(condition_numbers)) + 1 + min([49, smallest_de_analysis - 1])
    markers = create_condition_number_matrix(de_analysis_adjusted, sc_data, optimal_condition_number, annotations).index

    return pd.Series(markers)


def build_rectangle_signatures(
    sc_counts: pd.DataFrame, annotations: pd.Series, with_recursive_step=True
) -> RectangleSignatureResult:
    """Builds rectangle signatures based on single-cell count data and annotations.

    Parameters
    ----------
    sc_counts
        The single-cell count data as a DataFrame. DataFrame should have the genes as rows and cell as columns.
    annotations
        The annotations corresponding to the single-cell count data. Series should have the corresponding annotations for each cell.
    with_recursive_step
        Indicates whether to include the recursive clustering step. Defaults to True.

    Returns
    -------
    The result of the rectangle signature analysis.
    """
    assert sc_counts is not None and annotations is not None

    pseudo_signature_counts = sc_counts.groupby(annotations.values, axis=1).sum()

    print("creating signature")
    biasfact = (pseudo_signature_counts > 0).sum(axis=0)
    biasfact = biasfact / biasfact.min()
    genes = get_limma_genes_condition(pseudo_signature_counts, sc_counts, annotations)
    pseudo_signature_counts = convert_to_cpm(pseudo_signature_counts)
    if not with_recursive_step or len(pseudo_signature_counts.columns) < 4:
        return RectangleSignatureResult(
            signature_genes=genes, pseudobulk_sig_cpm=pseudo_signature_counts, bias_factors=biasfact
        )

    print("creating clustered signature")
    linkage_matrix = create_linkage_matrix(pseudo_signature_counts.loc[genes])
    clusters = create_fclusters(pseudo_signature_counts.loc[genes], linkage_matrix)
    assignments = None
    clustered_signature = None
    clustered_biasfact = None
    clustered_genes = None
    if len(set(clusters)) > 2:
        assignments = get_fcluster_assignments(clusters, pseudo_signature_counts.columns)
        clustered_annotations = create_annotations_from_cluster_labels(
            assignments, annotations, pseudo_signature_counts
        )

        clustered_signature = sc_counts.groupby(clustered_annotations.values, axis=1).sum()
        clustered_biasfact = (clustered_signature > 0).sum(axis=0)
        clustered_biasfact = clustered_biasfact / clustered_biasfact.min()
        clustered_genes = get_limma_genes_condition(clustered_signature, sc_counts, clustered_annotations)
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
