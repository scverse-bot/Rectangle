import pandas as pd


class RectangleSignatureResult:
    """Represents the result of a rectangle signature analysis (Created by the method pp.build_rectangle_signatures).

    Parameters
    ----------
    signature_genes
        The signature genes as a pd.Series.
    bias_factors
        The mRNA bias factors associated with each cell type.
    pseudobulk_sig_cpm
        The pseudo bulk signature build from the single cell data, contains all genes. Normalized to CPM.
    clustered_pseudobulk_sig_cpm
        The  clustered pseudo bulk signature build from the single cell data, contains all genes. Normalized to CPM.
    clustered_bias_factors
        The bias factors associated with each cluster.
    cluster_assignments
        The assignments of signature cell-types to clusters, as a list of ints or strings. In the same order as the pseudobulk_sig_cpm columns.
    """

    def __init__(
        self,
        signature_genes: pd.Series,
        bias_factors: pd.Series,
        pseudobulk_sig_cpm: pd.DataFrame,
        clustered_pseudobulk_sig_cpm: pd.DataFrame = None,
        clustered_bias_factors: pd.Series = None,
        clustered_signature_genes: pd.Series = None,
        cluster_assignments: list[int or str] = None,
    ):
        self.signature_genes = signature_genes
        self.bias_factors = bias_factors
        self.pseudobulk_sig_cpm = pseudobulk_sig_cpm
        self.clustered_pseudobulk_sig_cpm = clustered_pseudobulk_sig_cpm
        self.clustered_bias_factors = clustered_bias_factors
        self.clustered_signature_genes = clustered_signature_genes
        self.assignments = cluster_assignments
