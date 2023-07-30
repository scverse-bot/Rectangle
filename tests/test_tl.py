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
def finotello_bulk(data_dir):
    bulk = pd.read_csv(data_dir / "finotello_bulk.csv", index_col=0, header=0)
    return bulk


@pytest.fixture
def dream_bulk(data_dir):
    bulk = pd.read_csv(data_dir / "dream.csv", index_col=0, header=0)
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
    sc_counts = sc_counts.astype("int")
    signature = rectangle.pp.build_rectangle_signatures(sc_counts, annotations, 0.2, 1, False)
    bulk, _, _ = quantiseq_data
    bulk = bulk.iloc[:, 11]
    pseudo_signature = signature.pseudobulk_sig_cpm
    sig = pseudo_signature.loc[signature.signature_genes]
    fractions = rectangle.tl.weighted_dampened_deconvolute(sig, bulk)
    biasfact = (pseudo_signature > 0).sum(axis=0)
    biasfact = biasfact / biasfact.min()
    result = rectangle.tl.correct_for_unknown_cell_content(bulk, pseudo_signature, fractions, biasfact)
    assert len(fractions) + 1 == len(result)
