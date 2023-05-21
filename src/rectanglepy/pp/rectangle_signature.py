import pandas as pd


class RectangleSignatureResult:
    """12Represents the result of a rectangle signature analysis.

    Parameters
    ----------
    signature
        The signature data as a DataFrame.
    bias_factors
        The bias factors associated with the signature data. Is already multiplied into the signature.
    pseudo_signature
        The pseudo signature data as a DataFrame. Is used for correction of unknown content in the deconvolution step.
    clustered_signature
        The clustered signature data as a DataFrame. Is only created when signature result is created with recursive step.
    clustered_bias_factors
        The bias factors associated with the clustered signature data.
    assignments
        The assignments of signature cell-types to clusters.
    """

    def __init__(
        self,
        signature: pd.DataFrame,
        bias_factors: pd.Series or None,
        pseudo_signature: pd.DataFrame,
        clustered_signature: pd.DataFrame = None,
        clustered_bias_factors: pd.Series = None,
        assignments: list[int or str] = None,
    ):
        self.signature = signature
        self.bias_factors = bias_factors
        self.pseudo_signature = pseudo_signature
        self.clustered_signature = clustered_signature
        self.clustered_bias_factors = clustered_bias_factors
        self.assignments = assignments
