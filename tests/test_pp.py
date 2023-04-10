from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import rectangle


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
def small_dwls_signature(data_dir):
    small_dwls_signature = pd.read_csv(data_dir / "dwls_model_small.csv", index_col=0)
    return small_dwls_signature


@pytest.fixture
def hao_signature(data_dir):
    hao_signature = pd.read_csv(data_dir / "dwls_signature_hao1.csv", index_col=0)
    return hao_signature


@pytest.fixture
def small_data_adata(small_data):
    sc_data, annotations = small_data
    annotations = pd.DataFrame(annotations, index=sc_data.columns)
    return ad.AnnData(X=sc_data.values.T, obs=annotations, var=pd.DataFrame(data=sc_data.index, index=sc_data.index))


@pytest.fixture
def mast_data(data_dir):
    mast = pd.read_csv(data_dir / "df_for_mast.csv", index_col=0)
    groups = np.array(np.repeat("cluster_other", 53).tolist() + np.repeat("cluster_1", 47).tolist())
    return mast, groups


@pytest.fixture
def mast_lr_test_result(data_dir):
    lr_result = pd.read_csv(data_dir / "lrTest.csv", index_col=0)
    return lr_result


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
    input, expected = log_data
    expected = expected[expected["log2_fc"] > 0.5]
    actual = rectangle.pp.stat_log2(input, group_v, 0.1)
    assert np.isclose(expected, actual, rtol=1e-05, atol=1e-05).all()


def test_create_data_for_mast(mast_data):
    mast, groups = mast_data
    result = rectangle.pp.create_data_for_mast(mast, groups)
    assert str(result.typeof) == "RTYPES.S4SXP"


def test_create_mast_zlm(mast_data):
    mast, groups = mast_data
    mast_data = rectangle.pp.create_data_for_mast(mast, groups)
    result = rectangle.pp.mast_zlm(mast_data)
    assert str(result.typeof) == "RTYPES.S4SXP"


def test_mast_lr_test(mast_data, mast_lr_test_result):
    mast, groups = mast_data
    mast_data = rectangle.pp.create_data_for_mast(mast, groups)
    zlm = rectangle.pp.mast_zlm(mast_data)
    result = rectangle.pp.mast_lr_test(zlm)
    np.isclose(mast_lr_test_result["value"], result["value"], rtol=1e-4, atol=1e-4).all()


def test_de_analysis(small_data):
    result = rectangle.pp.de_analysis(small_data[0], small_data[1])
    assert len(result) == 3


def test_signature_creation(small_data, small_dwls_signature):
    sc_data, annotations = small_data
    actual = rectangle.pp.signature_creation(sc_data, annotations).sort_index()
    expected = small_dwls_signature.sort_index()
    assert np.isclose(expected, actual, rtol=1e-05, atol=1e-05).all()


def test_create_linkage_matrix(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    assert len(linkage_matrix) == 10


def test_create_fclusters(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_fclusters(linkage_matrix, 10)
    # should trigger fallback to distance parameter
    clusters_max_1 = rectangle.pp.create_fclusters(linkage_matrix, 1)

    # cluster t-cells and  tregs
    assert list(clusters) == [3, 5, 5, 7, 1, 8, 5, 2, 4, 9, 6]
    # cluster only closest cell-types (t-cells)
    assert list(clusters_max_1) == [3, 5, 6, 8, 1, 9, 5, 2, 4, 10, 7]


def test_get_fcluster_assignments(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_fclusters(linkage_matrix, 10)
    assignments = rectangle.pp.get_fcluster_assignments(clusters, hao_signature.columns)
    assert assignments == ["Monocytes", 5, 5, "NK cells", "B cells", "pDC", 5, "Plasma cells", "mDC", "Platelet", "ILC"]


def test_create_annotations_from_cluster_labels(hao_signature):
    annotations = pd.Series(
        [
            "NK cells",
            "pDC",
            "Plasma cells",
            "ILC",
            "T cells CD8",
            "Platelet",
            "B cells",
            "mDC",
            "T cells CD4 conv",
            "Tregs",
            "Monocytes",
        ]
    )
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_fclusters(linkage_matrix, 10)
    assignments = rectangle.pp.get_fcluster_assignments(clusters, hao_signature.columns)
    annotations_from_cluster = rectangle.pp.create_annotations_from_cluster_labels(
        assignments, annotations, hao_signature
    )

    assert list(annotations_from_cluster) == [
        "NK cells",
        "pDC",
        "Plasma cells",
        "ILC",
        "5",
        "Platelet",
        "B cells",
        "mDC",
        "5",
        "5",
        "Monocytes",
    ]
