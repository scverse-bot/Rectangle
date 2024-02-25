import pandas as pd
from anndata import AnnData
from pandas import DataFrame
from pkg_resources import resource_stream

from .pp import RectangleSignatureResult, build_rectangle_signatures
from .tl import deconvolution


def rectangle(
    adata: AnnData,
    bulks: DataFrame,
    cell_type_col: str = "cell_type",
    *,
    layer: str = None,
    raw: bool = False,
    correct_mrna_bias: bool = True,
    optimize_cutoffs=True,
    p=0.015,
    lfc=1.5,
    balance_sc_data: bool = False,
    balance_number: int = 1500,
    n_cpus: int = None,
) -> tuple[DataFrame, RectangleSignatureResult]:
    r"""Builds rectangle signatures based on single-cell  count data and annotations.

    Parameters
    ----------
    adata
        The single-cell count data as a DataFrame. DataFrame must have the genes as index and cell identifier as columns. Each entry should be in raw counts.
    cell_type_col
        The annotations corresponding to the single-cell count data. Series data should have the cell identifier as index and the annotations as values.
    optimize_cutoffs
        Indicates whether to optimize the p-value and log fold change cutoffs using gridsearch. Defaults to True.
    p
        The p-value threshold for the DE analysis (only used if optimize_cutoffs is False).
    lfc
        The log fold change threshold for the DE analysis (only used if optimize_cutoffs is False).
    bulks
        todo
    n_cpus
        The number of cpus to use for the DE analysis. Defaults to the number of cpus available.

    layer
        todo
    raw
        todo
    correct_mrna_bias : bool, optional
        A flag indicating whether to correct for mRNA bias. Defaults to True.
    balance_sc_data : bool, optional
        A flag indicating whether to balance the single-cell data. Defaults to False.
    balance_number : int, optional
        The number of cells to balance the single-cell data to. Defaults to 1500. If cell number is less than this number it takes the original number of cells.

    Returns
    -------
    The result of the rectangle signature analysis which is of type RectangleSignatureResult.
    """
    assert isinstance(adata, AnnData), "adata must be an AnnData object"
    assert isinstance(bulks, DataFrame), "bulks must be a DataFrame"

    if bulks is not None:
        genes = list(set(bulks.columns) & set(adata.var_names))
        genes = sorted(genes)
        adata = adata[:, genes]
        bulks = bulks[genes]

    signatures = build_rectangle_signatures(
        adata,
        cell_type_col,
        bulks=bulks,
        optimize_cutoffs=optimize_cutoffs,
        layer=layer,
        raw=raw,
        p=p,
        lfc=lfc,
        n_cpus=n_cpus,
        balance_sc_data=balance_sc_data,
        balance_number=balance_number,
    )
    cell_fractions = deconvolution(signatures, bulks, correct_mrna_bias, n_cpus)

    return cell_fractions, signatures


def load_tutorial_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the single-cell count data, annotations, and bulk data from the tutorial.

    Returns
    -------
    The single-cell count data, annotations, and bulk data.
    """
    with resource_stream(__name__, "data/hao1_annotations_small.csv") as annotations_file:
        annotations = pd.read_csv(annotations_file, index_col=0)["0"]

    with resource_stream(__name__, "data/hao1_counts_small.csv") as counts_file:
        sc_counts = pd.read_csv(counts_file, index_col=0).astype("int")

    with resource_stream(__name__, "data/small_fino_bulks.csv") as bulks_file:
        bulks = pd.read_csv(bulks_file, index_col=0)

    return sc_counts.T, annotations, bulks.T
