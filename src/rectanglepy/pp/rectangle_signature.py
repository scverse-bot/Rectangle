import pandas as pd


class RectangleSignatureResult:
    """12Represents the result of a rectangle signature analysis.

    Parameters
    ----------
    signature_genes
        The signature data as a DataFrame.
    bias_factors
        The bias factors associated with the signature data. Is already multiplied into the signature.
    pseudobulk_sig_cpm
        The pseudo signature data as a DataFrame. Is used for correction of unknown content in the deconvolution step.
    clustered_pseudobulk_sig_cpm
        The clustered signature data as a DataFrame. Is only created when signature result is created with recursive step.
    clustered_bias_factors
        The bias factors associated with the clustered signature data.
    cluster_assignments
        The assignments of signature cell-types to clusters.
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
