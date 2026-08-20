"""Module with helper functionality for pandas

Functions
---------
replace_level_values(df, level, mapping)
    Replace values in a levels of a MultiIndex performantly.
"""
import typing as tp
from collections.abc import Mapping, Sequence, Hashable

import pandas as pd


PDType = tp.TypeVar("PDType", pd.DataFrame, pd.Series)

def replace_level_values(
        df: PDType,
        mapping: Mapping[tp.Any, Hashable],
        leveln: tp.Optional[int] = None,
        level_name: tp.Optional[str] = None
) -> PDType:
    """Replace values in a levels of a MultiIndex performantly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a MultiIndex.
    mapping : Mapping[Hashable, Hashable]
        Mapping of old values to new values. Keys that are not present in the
        level are ignored. Any values in the level that are not keys in the
        mapping are left unchanged.
    leveln : int, optional
        Index (0-based) of the level to replace values in. Either `leveln` or
        `level_name` must be specified.
    level_name : str, optional
        Name of the level to replace values in. Either `leveln` or `level_name`
        must be specified.

    Returns
    -------
    pd.DataFrame
        DataFrame with replaced values in the specified level.

    Raises
    ------
    TypeError
        If the DataFrame does not have a MultiIndex.
    ValueError
        If neither `leveln` nor `level_name` is specified, or if both are
        specified.
    IndexError
        If `leveln` is out of range.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("DataFrame must have a MultiIndex.")
    if leveln is None:
        if level_name is None:
            raise ValueError("Either `leveln` or `level_name` must be specified.")
        leveln = df.index.names.index(level_name)
    elif level_name is not None:
        raise ValueError("Only one of `leveln` or `level_name` may be specified.")
    else:
        level_name = df.index.names[leveln]

    # Check that the keys in `mapping` are unique, and raise a ValueError if
    # they are not.
    if len(mapping) != len(set(mapping)):
        raise ValueError("Keys in `mapping` must be unique.")

    # First get the level and map the values directly. If the resulting index
    # object has unique values, we can set that using `set_levels` and be done.
    level_index: pd.Index = df.index.levels[leveln]
    replace_map: dict[Hashable, Hashable] = {
        k: mapping.get(k, k) for k in level_index
    }
    new_level_index: pd.Index = level_index.map(replace_map)
    if new_level_index.is_unique:
        df = df.copy()
        assert isinstance(df.index, pd.MultiIndex)
        df.index = df.index.set_levels(new_level_index, level=level_name)
        return df
    # If the resulting index is not unique, we need to turn the level into a
    # Series, map the values of the seriers, and then set the level back to the
    # mapped values.
    level_df = df.reset_index(level_name)[[level_name]]
    level_df[level_name] = level_df[level_name].replace(dict(mapping))
    level_df = level_df.set_index(level_name, append=True)
    level_df = level_df.reorder_levels(df.index.names)
    df = df.copy()
    df.index = level_df.index
    return df

###END def replace_level_values
