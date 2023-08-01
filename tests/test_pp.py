from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rectanglepy as rectangle
from rectanglepy.pp.create_signature import (
    build_rectangle_signatures,
    convert_to_cpm,
    create_annotations_from_cluster_labels,
    create_fclusters,
    create_linkage_matrix,
    get_fcluster_assignments,
)
from rectanglepy.pp.deconvolution import (
    correct_for_unknown_cell_content,
    scale_weights,
    solve_dampened_wsl,
    weighted_dampened_deconvolute,
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


@pytest.fixture
def finotello_bulk(data_dir):
    bulk = pd.read_csv(data_dir / "finotello_bulk.csv", index_col=0, header=0)
    return bulk


@pytest.fixture
def dream_bulk(data_dir):
    bulk = pd.read_csv(data_dir / "dream.csv", index_col=0, header=0)
    return bulk


def test_create_linkage_matrix(hao_signature):
    linkage_matrix = create_linkage_matrix(hao_signature)
    assert len(linkage_matrix) == 10


def test_create_fclusters(hao_signature):
    linkage_matrix = create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_signature.create_fclusters(hao_signature, linkage_matrix)
    assert clusters == [2, 3, 3, 4, 1, 5, 3, 1, 2, 6, 3]


def test_get_fcluster_assignments(hao_signature):
    linkage_matrix = create_linkage_matrix(hao_signature)
    clusters = create_fclusters(hao_signature, linkage_matrix)
    assignments = get_fcluster_assignments(clusters, hao_signature.columns)
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
    linkage_matrix = create_linkage_matrix(hao_signature)
    clusters = create_fclusters(hao_signature, linkage_matrix)
    assignments = get_fcluster_assignments(clusters, hao_signature.columns)
    annotations_from_cluster = create_annotations_from_cluster_labels(assignments, annotations, hao_signature)

    assert list(annotations_from_cluster) == ["NK cells", "pDC", "1", "3", "3", "Platelet", "1", "2", "3", "3", "2"]


def test_convert_to_cpm(small_data):
    count_sc_data = small_data[0]
    cpm_sc_data = convert_to_cpm(count_sc_data)
    assert np.isclose(cpm_sc_data.iloc[1, 0], 179.69972)


def test_scale_weigths():
    weights = [1, 0]
    result = scale_weights(weights)
    assert (result == [np.Inf, 0]).all()


def test_solve_dampened_wsl(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 11
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    result = solve_dampened_wsl(signature, bulk)
    # evaluation metrics
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.92 and rsme < 0.015


def test_simple_weighted_dampened_deconvolution(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 11
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    result = weighted_dampened_deconvolute(signature, bulk)
    # evaluation metrics
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.92 and rsme < 0.015


def test_correct_for_unknown_cell_content(small_data, quantiseq_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    signature = build_rectangle_signatures(sc_counts, annotations, p=0.2, lfc=1, optimize_cutoffs=False)
    bulk, _, _ = quantiseq_data
    bulk = bulk.iloc[:, 11]
    pseudo_signature = signature.pseudobulk_sig_cpm
    sig = pseudo_signature.loc[signature.signature_genes]
    fractions = weighted_dampened_deconvolute(sig, bulk)
    biasfact = (pseudo_signature > 0).sum(axis=0)
    biasfact = biasfact / biasfact.min()
    result = correct_for_unknown_cell_content(bulk, pseudo_signature, fractions, biasfact)
    assert len(fractions) + 1 == len(result)
