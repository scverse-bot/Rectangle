import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects.packages as rpackages
import statsmodels.stats.multitest as multi
from anndata import AnnData
from pandas import DataFrame
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.cluster.hierarchy import fcluster, linkage


def make_pseudo_bulk_adata(adata: AnnData, group_size: int) -> AnnData:
    """Calculate the mean of N columns, where each column group has the same secondary index.

    Parameters
    ----------
    adata
        - anndata.AnnData: Input anndata.
    group_size
        - int: Number of columns to group together.

    Returns
    -------
        - anndata.AnnData: A AnnData with the mean of each data matrix column group.
    """
    assert False is adata.var.empty
    assert False is adata.obs.empty

    annotations = adata.obs.iloc[:, 0]
    sc_data = adata.to_df().T

    bulk_df, bulk_annotations = make_pseudo_bulk(sc_data, annotations, group_size)
    return ad.AnnData(X=bulk_df.values.T, obs=pd.DataFrame(bulk_annotations, index=bulk_df.columns), var=adata.var)


def make_pseudo_bulk(sc_data: pd.DataFrame, annotations: pd.Series, group_size=2):
    assert all(annotations.value_counts() >= group_size)
    sc_data.columns = [list(annotations), list(sc_data.columns)]
    sc_data = sc_data.sort_index(axis=1)
    sorted_annotations = annotations.sort_values()
    sorted_annotations.index = range(len(sorted_annotations))
    bulk_df = sc_data[sc_data.columns[:0]].copy()

    for annotation in sorted_annotations.unique():
        first_index = sorted_annotations.index[sorted_annotations == annotation][0]
        last_index = sorted_annotations.index[sorted_annotations == annotation][-1]
        for x in range(first_index, last_index + 1, group_size):
            grouped_columns = sc_data.iloc[:, x : x + group_size]
            column_name = "_".join([c[1] for c in grouped_columns.columns])
            bulk_df[(annotation, column_name)] = grouped_columns.mean(axis=1)

    annotations_bulk = [str(x[0]) for x in bulk_df.columns]
    bulk_df.columns = [str(x[1]) for x in bulk_df.columns]

    return bulk_df, annotations_bulk


def convert_to_cpm(count_sc_data):
    return count_sc_data * 1e6 / np.sum(count_sc_data)


def mean_in_log2space(values, pseudo_count):
    return np.log2((2**values - pseudo_count).mean() + pseudo_count)


def check_mast_install():
    if not rpackages.isinstalled("BiocManager"):
        print("BiocManager not installed")
    if not rpackages.isinstalled("MAST"):
        print("MAST is not installed")


def create_data_for_mast(counts, groups):
    check_mast_install()
    mast = importr("MAST")
    # can't get to make the conversion work with sparse matrices, so we convert to dense
    if pd.api.types.is_sparse(counts.iloc[:, 0]):
        counts = counts.sparse.to_dense()
    with localconverter(robjects.default_converter + pandas2ri.converter):
        f_data_r = robjects.conversion.get_conversion().py2rpy(pd.DataFrame(data={"primerid": counts.index}))
        c_data_r = robjects.conversion.get_conversion().py2rpy(
            pd.DataFrame(data={"wellKey": counts.T.index, "Population": groups, "ncells": 1})
        )
        robjects.r.assign("counts.data", robjects.conversion.get_conversion().py2rpy(counts))
    return mast.FromMatrix(robjects.r("data.matrix(counts.data)"), c_data_r, f_data_r)


def mast_lr_test(zlm_output):
    mast = importr("MAST")
    robjects.r.assign("zlm.lr", mast.lrTest(zlm_output, "Population"))
    robjects.r.assign("zlm.lr", robjects.r('reshape2::melt(zlm.lr[, , "Pr(>Chisq)"])'))
    with localconverter(robjects.default_converter + pandas2ri.converter):
        zlm_lr_df = robjects.conversion.get_conversion().rpy2py(
            robjects.r('zlm.lr[which(zlm.lr$test.type == "hurdle"), ]')
        )
    return zlm_lr_df


def create_lr_test_df(annotation, de, lr_pval):
    lr_test_df = lr_pval.merge(de, left_on="primerid", right_index=True)
    lr_test_df.columns = [
        "gene",
        "test_type",
        "p_val",
        "log_mean_" + "cluster_other",
        "log_mean_" + annotation,
        "log2_fc",
    ]
    return lr_test_df


def lr_test(annotation, log2_results, values_df, group):
    zlm_output = mast_zlm(create_data_for_mast(values_df, group))
    return create_lr_test_df(annotation, log2_results, mast_lr_test(zlm_output))


def de_analysis(sc_data: pd.DataFrame, annotations: pd.Series):
    test_results = {}
    pseudo_count = 0.1
    data_log2 = np.log2(sc_data + pseudo_count)
    for annotation in np.unique(annotations):
        data_log2 = data_log2.loc[:, annotations != annotation].join(data_log2.loc[:, annotations == annotation])
        group_log2 = np.array([1 if x else 0 for x in sorted(annotations == annotation)])
        log2_results = stat_log2(data_log2, group_log2, pseudo_count)
        genes_list = log2_results.index.tolist()
        groups = np.array([annotation if x else "cluster_other" for x in group_log2])
        test_result = lr_test(annotation, log2_results, data_log2.loc[genes_list], groups)
        test_results[annotation] = test_result
    return test_results


def stat_log2(values_df, group, pseudo_count):
    log_cutt_off = 0.5
    values_df_t = values_df.T
    values_df_t["group"] = group
    log_mean_r = values_df_t.groupby("group").apply(lambda x: mean_in_log2space(x, pseudo_count)).T
    log_mean_r.columns = ["log_g_zero", "log_g_one"]
    log_mean_r["log2_fc"] = log_mean_r["log_g_one"] - log_mean_r["log_g_zero"]
    log_mean_r = log_mean_r[log_mean_r["log2_fc"] > log_cutt_off]
    return log_mean_r.drop("group")


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


def mast_zlm(mast_data):
    mast = importr("MAST")
    return mast.zlm(robjects.Formula("~Population"), mast_data, method="bayesglm", ebayes=True)


def create_linkage_matrix(signature: pd.DataFrame):
    method = "complete"
    metric = "euclidean"
    return linkage((np.log(signature + 1)).T, method=method, metric=metric)


def create_fclusters(linkage_matrix, maxclust) -> list[int]:
    clusters = fcluster(linkage_matrix, criterion="maxclust", t=maxclust)
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


def signature_creation(sc_data: pd.DataFrame, annotations: pd.Series, do_cpm_conversion=True) -> pd.DataFrame:
    print("signature creation")
    p_cutoff = 0.01
    # remove unexpressed genes
    sc_data = sc_data.loc[~(sc_data == 0).all(axis=1)]

    if do_cpm_conversion:
        sc_data = convert_to_cpm(sc_data)

    de_analysis_results = de_analysis(sc_data, annotations)
    de_analysis_adjusted = {}

    for annotation in annotations.unique():
        de_analysis_result = de_analysis_results[annotation]
        de_analysis_result["p_val_adjusted"] = multi.multipletests(de_analysis_result["p_val"], method="fdr_bh")[1]
        de_analysis_adjusted[annotation] = de_analysis_result[
            (de_analysis_result["p_val_adjusted"] < p_cutoff) & (de_analysis_result["log2_fc"] > 0.5)
        ]

    condition_number_matrices = create_condition_number_matrices(de_analysis_adjusted, sc_data, annotations)
    condition_numbers = [np.linalg.cond(np.linalg.qr(x)[1], 1) for x in condition_number_matrices]
    smallest_de_analysis = min([len(de_analysis_adjusted[annotation]) for annotation in de_analysis_adjusted])
    optimal_condition_number = condition_numbers.index(min(condition_numbers)) + 1 + min([49, smallest_de_analysis - 1])
    return create_condition_number_matrix(de_analysis_adjusted, sc_data, optimal_condition_number, annotations)


def calculate_bias_factors(sc_data, annotations, signature):
    sc_bias_factor = sc_data.gt(0).sum(axis=0)
    bias_factors = [np.mean(sc_bias_factor[[annotation == x for x in annotations]]) for annotation in signature.columns]
    bias_factors /= np.min(bias_factors)
    return bias_factors


def build_rectangle_signatures_adata(
    adata: AnnData, convert_to_cpm=True, with_recursive_step=True, calculate_bias=True
) -> pd.DataFrame:
    return build_rectangle_signatures(
        adata.to_df().T, adata.obs.iloc[:, 0], convert_to_cpm, with_recursive_step, calculate_bias
    )


def build_rectangle_signatures(
    sc_counts: pd.DataFrame, annotations: pd.Series, convert_to_cpm=True, with_recursive_step=True, calculate_bias=True
) -> DataFrame | tuple[DataFrame, DataFrame, list[int | str]]:
    assert sc_counts is not None and annotations is not None

    print("creating signature")
    signature = signature_creation(sc_counts, annotations, convert_to_cpm)

    if calculate_bias:
        bias_factors = calculate_bias_factors(sc_counts, annotations, signature)
        signature = signature * bias_factors

    if not with_recursive_step:
        return signature

    print("creating clustered signature")
    linkage_matrix = create_linkage_matrix(signature)
    clusters = create_fclusters(linkage_matrix, len(signature.columns) - 1)
    assignments = get_fcluster_assignments(clusters, signature.columns)
    clustered_annotations = create_annotations_from_cluster_labels(assignments, annotations, signature)

    clustered_signature = signature_creation(sc_counts, clustered_annotations, convert_to_cpm)
    if calculate_bias:
        clustered_bias_factors = calculate_bias_factors(sc_counts, clustered_annotations, clustered_signature)
        clustered_signature = clustered_signature * clustered_bias_factors

    return (signature, clustered_signature, assignments)
