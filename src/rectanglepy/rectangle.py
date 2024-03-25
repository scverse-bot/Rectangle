import pandas as pd
from anndata import AnnData
from loguru import logger
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
    subsample: bool = False,
    sample_size: int = 1500,
    consensus_runs: int = 1,
    correct_mrna_bias: bool = True,
    optimize_cutoffs=True,
    p=0.015,
    lfc=1.5,
    n_cpus: int = None,
) -> tuple[DataFrame, RectangleSignatureResult]:
    r"""All in one deconvolution method. Creates signatures and deconvolutes the bulk data. Has options for subsampling and consensus runs.

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
    consensus_runs : int
        The number of consensus runs to perform. Defaults to 1 for a singular deconvolution run. Consensus runs are performed by subsampling the single-cell data and running the analysis multiple times. The results are then aggregated.
    optimize_cutoffs
        Indicates whether to optimize the p-value and log fold change cutoffs using gridsearch. Defaults to True.
    p
        The p-value threshold for the DE analysis (only used if optimize_cutoffs is False).
    lfc
        The log fold change threshold for the DE analysis (only used if optimize_cutoffs is False).
    n_cpus
        The number of cpus to use for the DE analysis. Defaults to the number of cpus available.
    correct_mrna_bias : bool
        A flag indicating whether to correct for mRNA bias. Defaults to True.

    Returns
    -------
    DataFrame : The estimated cell fractions.
    RectangleSignatureResult : The result of the rectangle signature analysis.
    """
    assert isinstance(adata, AnnData), "adata must be an AnnData object"
    assert isinstance(bulks, DataFrame), "bulks must be a DataFrame"

    if bulks is not None:
        genes = list(set(bulks.columns) & set(adata.var_names))
        genes = sorted(genes)
        adata = adata[:, genes]
        bulks = bulks[genes]

    if consensus_runs > 1:
        logger.info(f"Running {consensus_runs} consensus runs with subsample size {sample_size}")
        subsample = True

    estimations = []
    most_recent_signatures = None

    for _i in range(consensus_runs):
        logger.info(f"Running consensus run {_i + 1} of {consensus_runs}")
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
            subsample=subsample,
            sample_size=sample_size,
        )
        cell_fractions = deconvolution(signatures, bulks, correct_mrna_bias, n_cpus)
        estimations.append(cell_fractions)
        most_recent_signatures = signatures

    return pd.concat(estimations).groupby(level=0).median(), most_recent_signatures


def load_tutorial_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the single-cell count data, annotations, and bulk data from the tutorial.

    Returns
    -------
    The single-cell count data, annotations, and bulk data.
    """
    with resource_stream(__name__, "data/hao1_annotations_small.zip") as annotations_file:
        annotations = pd.read_csv(annotations_file, index_col=0, compression="zip")["0"]

    with resource_stream(__name__, "data/hao1_counts_small.zip") as counts_file:
        sc_counts = pd.read_csv(counts_file, index_col=0, compression="zip").astype("int")

    with resource_stream(__name__, "data/small_fino_bulks.zip") as bulks_file:
        bulks = pd.read_csv(bulks_file, index_col=0, compression="zip")

    return sc_counts.T, annotations, bulks.T
