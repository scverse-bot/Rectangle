from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from src import rectangle


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def small_data(data_dir):
    sc_data = pd.read_csv(data_dir / "sc_object_small.csv", index_col=0)
    annotations = list(pd.read_csv(data_dir / "cell_annotations_small.txt", header=None, index_col=0).index)
    annotations = pd.Series(annotations, index=sc_data.columns)
    return sc_data, annotations


@pytest.fixture
def small_data_adata(small_data):
    sc_data, annotations = small_data
    annotations = pd.DataFrame(annotations, index=sc_data.columns)
    return ad.AnnData(X=sc_data.values.T, obs=annotations, var=pd.DataFrame(data=sc_data.index, index=sc_data.index))


@pytest.fixture
def log_data(data_dir):
    input = pd.read_csv(data_dir / "data_stat_log2.csv", index_col=0)
    expected = pd.read_csv(data_dir / "result_stat_log2.csv", index_col=0)
    return input, expected


# TODO: Add tests with sparse data


def test_psuedo_bulk_creation(small_data):
    group_size = 3
    sc_data, annotations = small_data
    bulk, bulk_annotations = rectangle.pp.make_pseudo_bulk(sc_data, annotations, group_size)

    assert abs(len(bulk_annotations) - len(annotations) // group_size) <= 1
    assert abs(bulk.values.sum() * 3 - sc_data.values.sum()) < 50000


def test_pseudo_bulk_creation_adata(small_data_adata, small_data):
    group_size = 3
    sc_data, annotations = small_data
    adata = small_data_adata
    result = rectangle.pp.make_pseudo_bulk_adata(adata, group_size)
    bulk, bulk_annotations = result.to_df().T, result.obs.index

    assert abs(len(bulk_annotations) - len(annotations) // group_size) <= 1
    assert abs(bulk.values.sum() * 3 - sc_data.values.sum()) < 50000


def test_stat_log2(log_data):
    group_v = np.append(np.repeat([0], 53), np.repeat(1, 47))
    pseudo_count = 0.1
    input, expected = log_data
    actual = rectangle.pp.stat_log2(input, group_v, pseudo_count)
    assert np.isclose(expected, actual, rtol=1e-05, atol=1e-05).all()
