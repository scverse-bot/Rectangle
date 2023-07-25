import pandas as pd
from src import rectanglepy
from src.rectanglepy.pp.create_signature import generate_deseq2, optimize_parameters


def rectangle(
    sc_data: pd.DataFrame, annotations: pd.Series, bulks: pd.DataFrame, optimize_cutoffs: bool = True, p=0.02, lfc=1.0
):
    consistency_check(sc_data, annotations, bulks)

    bulks, sc_data = reduce_to_common_genes(bulks, sc_data)

    pseudo_signature_counts = sc_data.groupby(annotations.values, axis=1).sum()

    de_results = generate_deseq2(pseudo_signature_counts)

    if optimize_cutoffs:
        p, lfc = optimize_parameters(sc_data, annotations, pseudo_signature_counts, de_results)

    signature_result = rectanglepy.pp.build_rectangle_signatures(sc_data, annotations, p, lfc, optimize_cutoffs)

    estimations = bulks.apply(lambda x: rectanglepy.tl.recursive_deconvolute(signature_result, x), axis=0)

    return estimations


def consistency_check(sc_data: pd.DataFrame, annotations: pd.Series, bulks: pd.DataFrame):
    assert sc_data is not None and annotations is not None and bulks is not None
    assert isinstance(sc_data, pd.DataFrame)
    assert isinstance(annotations, pd.Series)
    assert isinstance(bulks, pd.DataFrame)
    assert len(sc_data.columns) == len(annotations.index)


def reduce_to_common_genes(bulks: pd.DataFrame, sc_data: pd.DataFrame):
    genes = list(set(bulks.index) & set(sc_data.index))
    sc_data = sc_data.loc[genes].sort_index()
    bulks = bulks.loc[genes].sort_index()
    return bulks, sc_data
