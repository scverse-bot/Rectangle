import pandas as pd
from anndata import AnnData

import rectanglepy.rectangle
from rectanglepy.pp import RectangleSignatureResult


def test_load_tutorial_data():
    sc_data, annotations, bulks = rectanglepy.load_tutorial_data()
    assert isinstance(sc_data, pd.DataFrame)
    assert isinstance(annotations, pd.Series)
    assert isinstance(bulks, pd.DataFrame)


def test_rectangle():
    sc_data, annotations, bulks = rectanglepy.load_tutorial_data()
    sc_data_adata = AnnData(sc_data, obs=annotations.to_frame(name="cell_type"))

    result = rectanglepy.rectangle(sc_data_adata, bulks)

    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], RectangleSignatureResult)


def test_rectangle_consensus():
    sc_data, annotations, bulks = rectanglepy.load_tutorial_data()
    sc_data_adata = AnnData(sc_data, obs=annotations.to_frame(name="cell_type"))

    result = rectanglepy.rectangle(
        sc_data_adata, bulks, optimize_cutoffs=False, p=0.5, lfc=0.0, consensus_runs=2, sample_size=50
    )

    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], RectangleSignatureResult)
