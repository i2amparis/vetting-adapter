"""Names and identifiers for DataFrame columns and related labels."""
import enum



class CTCol(enum.StrEnum):
    """Column names for DataFrames received by `CriterionTargetRangeOutput`."""
    INRANGE = 'in_range'
    DISTANCE = 'distance'
    VALUE = 'value'
###END enum CTCol


class SummaryColumnSource(enum.Enum):
    """Specifies where to get the column names of summary DataFrames from."""
    DICT_KEYS = enum.auto()
    CRITERIA_NAMES = enum.auto()
###END enum class SummaryColumnSource
