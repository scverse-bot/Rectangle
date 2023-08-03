import pandas as pd
from loguru import logger

from rectanglepy.pp import build_rectangle_signatures, recursive_deconvolute


def rectangle(
    sc_data: pd.DataFrame,
    annotations: pd.Series,
    bulks: pd.DataFrame,
    *,
    optimize_cutoffs: bool = True,
    p=0.02,
    lfc=1.0,
) -> pd.DataFrame:
    """Run rectangle on a dataset.

    Parameters
    ----------
    sc_data
        todo
    annotations
        todo
    bulks
        todo
    optimize_cutoffs
        todo
    p
        todo
    lfc
        todo.



    """
    _consistency_check(sc_data, annotations, bulks)

    bulks, sc_data = _reduce_to_common_genes(bulks, sc_data)

    signature_result = build_rectangle_signatures(sc_data, annotations, p=p, lfc=lfc, optimize_cutoffs=optimize_cutoffs)

    estimations = bulks.apply(lambda x: recursive_deconvolute(signature_result, x), axis=0)

    return estimations


def _consistency_check(sc_data: pd.DataFrame, annotations: pd.Series, bulks: pd.DataFrame):
    assert sc_data is not None and annotations is not None and bulks is not None
    assert isinstance(sc_data, pd.DataFrame)
    assert isinstance(annotations, pd.Series)
    assert isinstance(bulks, pd.DataFrame)
    assert len(sc_data.columns) == len(annotations.index)


def _reduce_to_common_genes(bulks: pd.DataFrame, sc_data: pd.DataFrame):
    genes = list(set(bulks.index) & set(sc_data.index))
    logger.info(f"Reducing bulks and sc data to {len(genes)} common genes")
    sc_data = sc_data.loc[genes].sort_index()
    bulks = bulks.loc[genes].sort_index()
    return bulks, sc_data
