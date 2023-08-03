from pathlib import Path

import pandas as pd
import pytest

from rectanglepy.pp.create_signature import (
    _generate_estimated_fractions,
    _generate_pseudo_bulks,
    _optimize_parameters,
    _run_deseq2,
)
from rectanglepy.tl.rectangle import rectangle


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
def finotello_bulk(data_dir):
    bulk = pd.read_csv(data_dir / "finotello_bulk.csv", index_col=0, header=0)
    return bulk


def test_generate_pseudo_bulks(small_data):
    sc_data, annotations, bulk = small_data
    bulks, real_fractions = _generate_pseudo_bulks(sc_data, annotations)

    assert bulks.shape == (1000, 30)
    assert real_fractions.shape == (3, 30)


def test_generate_estimated_fractions(small_data):
    sc_data, annotations, bulk = small_data
    sc_data = sc_data.astype(int)
    bulks, real_fractions = _generate_pseudo_bulks(sc_data, annotations)
    pseudo_signature_counts = sc_data.groupby(annotations.values, axis=1).sum()
    de_results = _run_deseq2(pseudo_signature_counts)
    estimated_fractions = _generate_estimated_fractions(
        pseudo_signature_counts, bulks, 0.9, 0.1, de_results, sc_data, annotations
    )

    assert estimated_fractions.shape == (3, 30)


def test_optimize_parameters(small_data):
    sc_data, annotations, bulk = small_data
    sc_data = sc_data.astype(int)
    pseudo_signature_counts = sc_data.groupby(annotations.values, axis=1).sum()
    de_results = _run_deseq2(pseudo_signature_counts)
    optimized_parameters = _optimize_parameters(sc_data, annotations, pseudo_signature_counts, de_results)

    assert 0.018 <= optimized_parameters[0] <= 0.023
    assert 0.8 <= optimized_parameters[1] <= 1.2


def test_rectangle(small_data):
    sc_data, annotations, bulk = small_data
    sc_data = sc_data.astype(int)
    estimations = rectangle(sc_data, annotations, bulk)

    assert estimations.shape == (4, 8)
