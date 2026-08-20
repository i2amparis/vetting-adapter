"""Module for definitin dimension (column) names for IAM datasets."""
from typing import NewType
from dataclasses import dataclass


class IamDimName(str):
    """String type used for IAM dimension names."""
    ...

@dataclass(kw_only=True, frozen=True)
class IamDimNames:
    """Dimension (column) names used by `pyam.IamDataFrame` objects."""
    MODEL: IamDimName = IamDimName('model')
    SCENARIO: IamDimName = IamDimName('scenario')
    REGION: IamDimName = IamDimName('region')
    VARIABLE: IamDimName = IamDimName('variable')
    UNIT: IamDimName = IamDimName('unit')
    TIME: IamDimName = IamDimName('year')
###END class IamDim

DIM: IamDimNames = IamDimNames()
"""`IamDimNames` instance with default dimension names IamDataFrames."""


class UnknownDimensionNameError(ValueError):
    """Exception raised when an unknown dimension name is encountered."""
    ...
###END class UnknownDimensionNameError
