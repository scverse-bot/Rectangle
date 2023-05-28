import pickle
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


@pytest.fixture
def rectangle_signature(data_dir):
    rectangle_pickle = open(data_dir / "rectangle_limma", "rb")
    rectangle_signature = pickle.load(rectangle_pickle)
    return rectangle_signature


@pytest.fixture
def finotello_bulk(data_dir):
    bulk = pd.read_csv(data_dir / "finotello_bulk.csv", index_col=0, header=0)
    return bulk


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


def test_correct_for_unknown_cell_content(small_data, quantiseq_data):
    sc_counts, annotations, bulk = small_data
    signature = rectangle.pp.build_rectangle_signatures(sc_counts, annotations, False)
    bulk, _, _ = quantiseq_data
    bulk = bulk.iloc[:, 11]
    pseudo_signature = signature.pseudobulk_sig_cpm
    sig = pseudo_signature.loc[signature.signature_genes]
    fractions = rectangle.tl.weighted_dampened_deconvolute(sig, bulk)
    biasfact = (pseudo_signature > 0).sum(axis=0)
    biasfact = biasfact / biasfact.min()
    result = rectangle.tl.correct_for_unknown_cell_content(bulk, pseudo_signature, fractions, biasfact)
    assert len(fractions) + 1 == len(result)


def test_direct_deconvolute(rectangle_signature, finotello_bulk):
    bulk = finotello_bulk
    genes = rectangle_signature.signature_genes
    cpm_sig = rectangle_signature.pseudobulk_sig_cpm
    bias_factors = rectangle_signature.bias_factors
    direct_signature = cpm_sig.loc[genes] * bias_factors
    pseudo_sig = cpm_sig
    fractions = rectangle.tl.direct_deconvolute(direct_signature, bulk.iloc[:, 0], pseudo_sig, bias_factors)
    assert np.isclose(
        list(fractions),
        [
            0.0527192678982,
            0.002228634649488,
            0.28924320054,
            0.0211645570266,
            0.05338400303,
            0.0015461113220485,
            0.284169098555,
            0.239363407110,
            0.0048853574171,
            0.047282611465,
            0.00401375097792,
            0.0,
        ],
    ).all()


def test_recursive_deconvolute(rectangle_signature, finotello_bulk):
    bulk = finotello_bulk
    fractions = rectangle.tl.recursive_deconvolute(rectangle_signature, bulk.iloc[:, 0])
    assert np.isclose(
        list(fractions),
        [
            0.05092270317630736,
            0.0016720288310033168,
            0.28516997370978825,
            0.020415534530539167,
            0.05070825067693828,
            0.0014186546817250316,
            0.2662479872682521,
            0.22604376296776768,
            0.003952511070770494,
            0.0,
            0.048021807136958916,
            0.003623706964962541,
        ],
    ).all()
