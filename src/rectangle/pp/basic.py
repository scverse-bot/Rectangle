import anndata as ad
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

    annotations = list(adata.obs.iloc[:, 0])
    sc_data = pd.DataFrame(data=adata.X.T, columns=adata.obs.index, index=list(adata.var.iloc[:, 0]))

    bulk_df, bulk_annotations = _make_pseudo_bulk(sc_data, annotations, group_size)
    return ad.AnnData(X=bulk_df.values.T, obs=pd.DataFrame(bulk_annotations, index=bulk_df.columns), var=adata.var)


def _make_pseudo_bulk(sc_data: pd.DataFrame, annotations: list[str], group_size=2):
    assert all(annotations.count(x) > group_size for x in set(annotations))
    sc_data.columns = [list(annotations), list(sc_data.columns)]
    sc_data = sc_data.sort_index(axis=1)
    sorted_annotations = sorted(annotations)
    bulk_df = sc_data[sc_data.columns[:0]].copy()

    for annotation in sorted(set(sorted_annotations)):
        first_index = sorted_annotations.index(annotation)
        last_index = len(sorted_annotations) - sorted_annotations[::-1].index(annotation) - 1
        for x in range(first_index, last_index + 1, group_size):
            grouped_columns = sc_data.iloc[:, x : x + group_size]
            column_name = "_".join([c[1] for c in grouped_columns.columns])
            bulk_df[(annotation, column_name)] = grouped_columns.mean(axis=1)

    annotations_bulk = [str(x[0]) for x in bulk_df.columns]
    bulk_df.columns = [str(x[1]) for x in bulk_df.columns]

    return bulk_df, annotations_bulk
