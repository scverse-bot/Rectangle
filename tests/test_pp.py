from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rectanglepy as rectangle
from rectanglepy.pp.create_signature import (
    _create_annotations_from_cluster_labels,
    _create_fclusters,
    _create_linkage_matrix,
    _get_fcluster_assignments,
    build_rectangle_signatures,
)
from rectanglepy.pp.deconvolution import (
    _calculate_dwls,
    _scale_weights,
    correct_for_unknown_cell_content,
    solve_qp,
)


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def small_data(data_dir):
    sc_data = pd.read_csv(data_dir / "sc_object_small.csv", index_col=0)
    annotations = list(pd.read_csv(data_dir / "cell_annotations_small.txt", header=None, index_col=0).index)
    annotations = pd.Series(annotations, index=sc_data.columns)
    bulk = pd.read_csv(data_dir / "bulk_small.csv", index_col=0)
    return sc_data, annotations, bulk


@pytest.fixture
def hao_signature(data_dir):
    hao_signature = pd.read_csv(data_dir / "dwls_signature_hao1.csv", index_col=0)
    return hao_signature


@pytest.fixture
def quantiseq_data(data_dir):
    signature = pd.read_csv(data_dir / "TIL10_signature.txt", index_col=0, sep="\t")
    bulk = pd.read_csv(data_dir / "quanTIseq_SimRNAseq_mixture_small.txt", index_col=0, sep="\t")
    fractions = pd.read_csv(data_dir / "quanTIseq_SimRNAseq_read_fractions_small.txt", index_col=0, sep="\t")
    fractions = fractions.iloc[:, :-1]
    return bulk, fractions, signature


def test_create_linkage_matrix(hao_signature):
    linkage_matrix = _create_linkage_matrix(hao_signature)
    assert len(linkage_matrix) == 10


def test_create_fclusters(hao_signature):
    linkage_matrix = _create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_signature._create_fclusters(hao_signature, linkage_matrix)
    assert clusters == [3, 4, 4, 6, 1, 7, 4, 2, 3, 8, 5]


def test_get_fcluster_assignments(hao_signature):
    linkage_matrix = _create_linkage_matrix(hao_signature)
    clusters = _create_fclusters(hao_signature, linkage_matrix)
    assignments = _get_fcluster_assignments(clusters, hao_signature.columns)
    assert assignments == [3, 4, 4, "NK cells", "B cells", "pDC", 4, "Plasma cells", 3, "Platelet", "ILC"]


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
    linkage_matrix = _create_linkage_matrix(hao_signature)
    clusters = _create_fclusters(hao_signature, linkage_matrix)
    assignments = _get_fcluster_assignments(clusters, hao_signature.columns)
    annotations_from_cluster = _create_annotations_from_cluster_labels(assignments, annotations, hao_signature)

    assert list(annotations_from_cluster) == [
        "NK cells",
        "pDC",
        "Plasma cells",
        "ILC",
        "4",
        "Platelet",
        "B cells",
        "3",
        "4",
        "4",
        "3",
    ]


def test_scale_weigths():
    weights = [1, 0]
    result = _scale_weights(weights)
    assert (result == [np.Inf, 0]).all()


def test_solve_dampened_wsl(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 11
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    result = solve_qp(signature, bulk)
    # evaluation metrics
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.92 and rsme < 0.015


def test_simple_weighted_dampened_deconvolution(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 11
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    result = _calculate_dwls(signature, bulk)
    # evaluation metrics
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.85 and rsme < 0.015


def test_build_rectangle_signatures(small_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    results = build_rectangle_signatures(sc_counts, annotations, p=0.2, lfc=1, optimize_cutoffs=False)
    assert results.assignments is None  # should not cluster
    assert len(results.signature_genes) > 0


def test_correct_for_unknown_cell_content(small_data, quantiseq_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    signature = build_rectangle_signatures(sc_counts, annotations, p=0.9, lfc=0.1, optimize_cutoffs=False)
    bulk, _, _ = quantiseq_data
    bulk = bulk.iloc[:, 11]
    pseudo_signature = signature.pseudobulk_sig_cpm
    sig = pseudo_signature.loc[signature.signature_genes]
    fractions = _calculate_dwls(sig, bulk)
    biasfact = (pseudo_signature > 0).sum(axis=0)
    biasfact = biasfact / biasfact.min()
    result = correct_for_unknown_cell_content(bulk, pseudo_signature, fractions, biasfact)
    assert len(fractions) + 1 == len(result)
