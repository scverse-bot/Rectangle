import pandas as pd
from loguru import logger

from rectanglepy.pp import RectangleSignatureResult, build_rectangle_signatures, deconvolute
from rectanglepy.pp.create_signature import _reduce_to_common_genes


def rectangle(
    sc_data: pd.DataFrame,
    annotations: pd.Series,
    bulks: pd.DataFrame,
    *,
    optimize_cutoffs: bool = True,
    p=0.02,
    lfc=2.0,
    n_cpus: int = None,
) -> tuple[pd.DataFrame, RectangleSignatureResult]:
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
    n_cpus
        todo



    """
    _consistency_check(sc_data, annotations, bulks)
    sc_data = sc_data.loc[sc_data.sum(axis=1) > 10]

    signature_result = build_rectangle_signatures(
        sc_data, annotations, p=p, lfc=lfc, optimize_cutoffs=optimize_cutoffs, bulks=bulks, n_cpus=n_cpus
    )
    logger.info(f"Rectangle signature has {len(signature_result.signature_genes)} genes")

    bulks, sc_data = _reduce_to_common_genes(bulks, sc_data)

    estimations_data = {}
    for column_name, column_data in bulks.items():
        try:
            result = deconvolute(signature_result, column_data)
            estimations_data[column_name] = result
        except Exception as e:
            logger.error(f"An error occurred for column {column_name}: {e}")

    estimations_dataframe = pd.DataFrame(estimations_data)

    return estimations_dataframe, signature_result


def _consistency_check(sc_data: pd.DataFrame, annotations: pd.Series, bulks: pd.DataFrame):
    assert sc_data is not None and annotations is not None and bulks is not None
    assert isinstance(sc_data, pd.DataFrame)
    assert isinstance(annotations, pd.Series)
    assert isinstance(bulks, pd.DataFrame)
    assert len(sc_data.columns) == len(annotations.index)
