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
    bulk = pd.read_csv(data_dir / "bulk_small.csv", index_col=0)
    return sc_data, annotations, bulk


@pytest.fixture
def quantiseq_data(data_dir):
    signature = pd.read_csv(data_dir / "TIL10_signature.txt", index_col=0, sep="\t")
    bulk = pd.read_csv(data_dir / "quanTIseq_SimRNAseq_mixture_small.txt", index_col=0, sep="\t")
    fractions = pd.read_csv(data_dir / "quanTIseq_SimRNAseq_read_fractions_small.txt", index_col=0, sep="\t")
    fractions = fractions.iloc[:, :-1]
    return bulk, fractions, signature


@pytest.fixture
def small_dwls_signature(data_dir):
    small_dwls_signature = pd.read_csv(data_dir / "dwls_model_small.csv", index_col=0)
    return small_dwls_signature


def test_scale_weigths():
    weights = [1, 0]
    result = rectangle.tl.scale_weights(weights)
    assert (result == [np.Inf, 0]).all()


def test_solve_dampened_wsl(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 11
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    result = rectangle.tl.solve_dampened_wsl(signature, bulk)
    # evaluation metrics
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.92 and rsme < 0.015


def test_simple_weighted_dampened_deconvolution(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 11
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    result = rectangle.tl.weighted_dampened_deconvolute(signature, bulk)
    # evaluation metrics
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.92 and rsme < 0.015


def test_create_simple_pseudo_cpm_bulk_signature(small_data):
    sc_counts, annotations, bulk = small_data
    result = rectangle.tl.create_cpm_pseudo_signature(sc_counts, annotations)
    assert (len(result) == len(sc_counts)) & (result.iloc[1, 0] == 49.832037342450654)


def test_correct_for_unknown_cell_content(small_data, quantiseq_data):
    sc_counts, annotations, bulk = small_data
    signature = rectangle.pp.signature_creation(sc_counts, annotations)
    bulk, _, _ = quantiseq_data
    bulk = bulk.iloc[:, 11]
    fractions = rectangle.tl.weighted_dampened_deconvolute(signature, bulk)

    pseudo_signature = rectangle.tl.create_cpm_pseudo_signature(sc_counts, annotations)
    result = rectangle.tl.correct_for_unknown_cell_content(bulk, pseudo_signature, fractions)
    assert len(fractions) + 1 == len(result)


def test_direct_deconvolute(small_data):
    sc_counts, annotations, bulk = small_data
    signatures = rectangle.pp.build_rectangle_signatures(sc_counts, annotations, True)
    bulk = bulk.iloc[:, 1]
    rectangle.tl.recursive_deconvolute(signatures, bulk)


def test_recursive_deconvolute(small_data):
    sc_counts, annotations, bulk = small_data
    signatures = rectangle.pp.build_rectangle_signatures(sc_counts, annotations, True)
    bulk = bulk.iloc[:, 1]
    rectangle.tl.recursive_deconvolute(signatures, bulk)
