from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import rectangle

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_pseudo_bulk_creation_anndata():
    group_size = 3
    sc_data = pd.read_csv(TEST_DATA_DIR / "sc_object_small.csv", index_col=0)
    annotations = np.array(
        list(pd.read_csv(TEST_DATA_DIR / "cell_annotations_small.txt", header=None, index_col=0).index)
    )
    annotations = pd.DataFrame(annotations, index=sc_data.columns)
    adata = ad.AnnData(X=sc_data.values.T, obs=annotations, var=pd.DataFrame(data=sc_data.index, index=sc_data.index))
    result = rectangle.pp.make_pseudo_bulk_adata(adata, group_size)
    bulk, bulk_annotations = result.to_df().T, result.obs.index

    assert abs(len(bulk_annotations) - len(annotations) // group_size) <= 1
    assert abs(bulk.values.sum() * 3 - sc_data.values.sum()) < 50000
