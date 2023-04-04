import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData


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
    sc_data = pd.DataFrame(data=adata.X.T, columns=adata.obs.index, index=list(adata.var.iloc[:, 0]))

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
    return np.log2(np.mean(2**values - pseudo_count) + pseudo_count)


def stat_log2(values_df, group, pseudo_count):
    values_df_t = values_df.T
    values_df_t["group"] = group
    log2_mean_r = values_df_t.groupby("group").apply(lambda x: mean_in_log2space(x, pseudo_count)).T
    log2_mean_r.columns = ["log2.mean.group0", "log2.mean.group1"]
    log2_mean_r["log2_fc"] = log2_mean_r["log2.mean.group1"] - log2_mean_r["log2.mean.group0"]
    return log2_mean_r.drop("group")
