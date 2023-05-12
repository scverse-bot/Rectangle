from dataclasses import dataclass

import pandas as pd


@dataclass
class RectangleSignatureResult:
    """Represents the result of a rectangle signature analysis.

    Attributes:
    ----------
        signature (pd.DataFrame): The signature data as a DataFrame.
        bias_factors (pd.Series): The bias factors associated with the signature data.
        pseudo_signature (pd.DataFrame): The pseudo signature data as a DataFrame.
        clustered_signature (pd.DataFrame, optional): The clustered signature data as a DataFrame.
        clustered_bias_factors (pd.Series, optional): The bias factors associated with the clustered signature data.
        assignments (list[int | str], optional): The assignments of signature cell-types to clusters.

    Note:
    ----
        - The `signature` attribute is mandatory, while the other attributes are optional.
        - The `bias_factors` is already multiplied into the signature.
        - The `pseudo_signature` is used for correction of unknown content in the deconvolution step.
        - The `clustered_signature` is only created when signature result is created with recursive step.
    """

    signature: pd.DataFrame
    bias_factors: pd.Series
    pseudo_signature: pd.DataFrame
    clustered_signature: pd.DataFrame | None = None
    clustered_bias_factors: pd.Series | None = None
    assignments: list[int | str] | None = None
