from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rectanglepy as rectangle


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
def test_hao(data_dir):
    sc_data = pd.read_csv(data_dir / "hao1_matrix_test.csv", index_col=0)
    annotations = list(pd.read_csv(data_dir / "hao1_celltype_annotations_test.csv", header=0, index_col=0).index)
    annotations = pd.Series(annotations, index=sc_data.columns)
    return sc_data, annotations


@pytest.fixture
def hao_signature(data_dir):
    hao_signature = pd.read_csv(data_dir / "dwls_signature_hao1.csv", index_col=0)
    return hao_signature


def test_create_linkage_matrix(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    assert len(linkage_matrix) == 10


def test_create_fclusters(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_fclusters(hao_signature, linkage_matrix)
    assert clusters == [2, 3, 3, 4, 1, 5, 3, 1, 2, 6, 3]


def test_get_fcluster_assignments(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_fclusters(hao_signature, linkage_matrix)
    assignments = rectangle.pp.get_fcluster_assignments(clusters, hao_signature.columns)
    assert assignments == [2, 3, 3, "NK cells", 1, "pDC", 3, 1, 2, "Platelet", 3]


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
    clusters = rectangle.pp.create_fclusters(hao_signature, linkage_matrix)
    assignments = rectangle.pp.get_fcluster_assignments(clusters, hao_signature.columns)
    annotations_from_cluster = rectangle.pp.create_annotations_from_cluster_labels(
        assignments, annotations, hao_signature
    )

    assert list(annotations_from_cluster) == ["NK cells", "pDC", "1", "3", "3", "Platelet", "1", "2", "3", "3", "2"]


def test_build_rectangle_signatures_non_recursive(test_hao):
    sc_data, annotations = test_hao
    actual = rectangle.pp.build_rectangle_signatures(sc_data, annotations, False)
    assert actual.assignments is None
    assert actual.bias_factors[0] == 1.4037584525868847
    assert len(actual.signature_genes) == 1754


def test_build_rectangle_signatures_recursive(test_hao):
    sc_data, annotations = test_hao
    actual = rectangle.pp.build_rectangle_signatures(sc_data, annotations)
    assert actual.bias_factors[0] == 1.4037584525868847
    assert len(actual.signature_genes) == 1754
    assert len(actual.clustered_signature_genes) == 1188
    assert len(actual.clustered_bias_factors) == 6


def test_convert_to_cpm(small_data):
    count_sc_data = small_data[0]
    cpm_sc_data = rectangle.pp.convert_to_cpm(count_sc_data)
    assert np.isclose(cpm_sc_data.iloc[1, 0], 179.69972)
