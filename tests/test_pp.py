from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import rectanglepy as rectangle


def rds_to_df(filename, is_df=False):
    r_file = filename
    robjects.r(f"df_to_load <- readRDS('{r_file}')")
    r_df = robjects.r["df_to_load"]

    # Convert R dataframe to pandas dataframe
    with localconverter(robjects.default_converter + pandas2ri.converter):
        p_df = robjects.conversion.rpy2py(r_df)
    if is_df:
        p_df = pd.DataFrame(p_df)
        p_df.index = r_df.rownames
        p_df.columns = r_df.colnames

    return p_df


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
def hao_signature(data_dir):
    hao_signature = pd.read_csv(data_dir / "dwls_signature_hao1.csv", index_col=0)
    return hao_signature


def test_create_linkage_matrix(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    assert len(linkage_matrix) == 10


def test_create_fclusters(hao_signature):
    linkage_matrix = rectangle.pp.create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_signature.create_fclusters(hao_signature, linkage_matrix)
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


def test_convert_to_cpm(small_data):
    count_sc_data = small_data[0]
    cpm_sc_data = rectangle.pp.convert_to_cpm(count_sc_data)
    assert np.isclose(cpm_sc_data.iloc[1, 0], 179.69972)
