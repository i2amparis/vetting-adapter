"""Helper functions for the `pyam` package.

Functions
---------
make_consistent_units(df, match_df, unit_col='unit', match_dims=('variable',)) \
        -> pyam.IamDataFrame
    Make the units of an IamDataFrame consistent with another IamDataFrame.
    Converts units only for the variables where the units are different, does
    not convert a given unit for all variables the way
    `pyam.IamDataFrame.convert_unit` does. Only converts units for variables
    present in both IamDataFrames, and does not require both IamDataFrames to
    have the same regions, models, or scenarios, unless specified.
"""
import typing as tp
from collections.abc import Sequence

import pyam
import pandas as pd


TV = tp.TypeVar('TV')
def notnone(x: TV|None) -> TV:
    if x is None:
        raise ValueError('Value is None')
    return x
###END def notnone


def make_consistent_units(
        df: pyam.IamDataFrame,
        match_df: pyam.IamDataFrame,
        unit_col: str = 'unit',
        match_dims: Sequence[str] = ('model', 'scenario', 'region'),
        keep_meta: bool = True
) -> pyam.IamDataFrame:
    """Make the units of an IamDataFrame consistent with another IamDataFrame.

    Converts units only for the variables where the units are different, does
    not convert a given unit for all variables the way
    `pyam.IamDataFrame.convert_unit` does. Only converts units for variables
    present in both IamDataFrames, and does not require both IamDataFrames to
    have the same regions, models, or scenarios, unless specified.

    Parameters
    ----------
    df : pyam.IamDataFrame
        IamDataFrame to make the units consistent for.
    match_df : pyam.IamDataFrame
        IamDataFrame to match the units to.
    unit_col : str, optional
        Name of the dimension in the IamDataFrame that contains the units.
        Optional, defaults to 'unit'.
    keep_meta : bool, optional
        Whether to keep the metadata of `df` when converting units. Optional,
        defaults to True.
    match_dims : Sequence[str], optional
        Only used in the case that `match_df` has more than one unit for a
        single variable. In that case, this parameter specifies the dimensions
        that `df` and `match_df` will be matched on to select which unit to
        convert to. `match_df` must have only one unit per variable for a given
        combination of values for the dimensions specified in this parameter, or
        a `ValueError` will be raised. `variable` is always implied. Optional,
        defaults to all dimensions except the time dimension (`year`) and the
        `unit` dimension itself, i.e., `('model', 'scenario', 'region')`.

    Returns
    -------
    pyam.IamDataFrame
        IamDataFrame with units consisetent with `match_df`.
    """
    # First use the `pyam.IamDataFrame.unit_mapping` property to get a list of
    # all variables in `df` that have different units from `match_df`.
    df_unit_mapping: dict[str, str|list[str]] = df.unit_mapping
    match_unit_mapping: dict[str, list[str]] = match_df.unit_mapping
    differing_vars: list[str] = [
        _var for _var, _unit in df_unit_mapping.items()
        if _unit != match_unit_mapping.get(_var, _unit)
    ]
    matching_df: pyam.IamDataFrame = notnone(
        df.filter(variable=differing_vars, keep=False)
    )
    converted_dfs: list[pyam.IamDataFrame] = []
    for _var in differing_vars:
        # If _var has a single unit in `match_df`, convert it to that unit using
        # `pyam.IamDataFrame.convert_unit`.
        target_unit: str|list[str] = match_unit_mapping[_var]
        source_units: str|list[str] = df_unit_mapping[_var]
        if isinstance(source_units, str):
            source_units = [source_units]
        for _source_unit in source_units:
            if isinstance(target_unit, str):
                converted_dfs.append(
                    notnone(
                        notnone(df.filter(variable=_var, unit=_source_unit)) \
                            .convert_unit(_source_unit, to=target_unit)
                    )
                )
            else:
                _source_unit_converted_dfs: list[pyam.IamDataFrame] = []
                for _target_unit in target_unit:
                    target_unit_df: pyam.IamDataFrame = notnone(
                        match_df.filter(variable=_var, unit=_target_unit)
                    )
                    _df_filter: dict[str, str] = {
                        _dim: getattr(target_unit_df, _dim)
                        for _dim in match_dims
                    }
                    _source_df: pyam.IamDataFrame = notnone(
                        df.filter(
                            variable=_var,
                            unit=_source_unit,
                            keep=True,
                            inplace=False,
                            **_df_filter
                        )
                    )
                    _source_unit_converted_dfs.append(
                        notnone(
                            _source_df.convert_unit(_source_unit,
                                                    to=_target_unit)
                        )
                    )
                # Need to check that the result has the same length as the
                # source df for the variable and unit
                _source_unit_converted_df_joined = pyam.concat(
                    _source_unit_converted_dfs
                )
                if len(_source_unit_converted_df_joined) != \
                        len(notnone(df.filter(variable=_var, unit=_source_unit))):
                    raise ValueError(
                        f'The lenght of the converted IamDataFrame for '
                        f'variable {_var} and unit {_source_unit} does not '
                        f'match the length of the source IamDataFrame for the '
                        f'same variable and unit. Probably this is because the '
                        f'unit in the target IamDataFrame is not unique for '
                        f'the combination of dimensions given in the '
                        f'match_dims parameter ({match_dims}).'
                    )
                converted_dfs.extend(_source_unit_converted_dfs)
    converted_data_series: pd.Series = pd.concat(
        [matching_df._data, *[_df._data for _df in converted_dfs]]
    )
    converted_df: pyam.IamDataFrame = pyam.IamDataFrame(converted_data_series) \
        if not keep_meta else \
            pyam.IamDataFrame(converted_data_series, meta=df.meta)
    return converted_df
###END def make_consistent_units


def as_pandas_series(
        df: pyam.IamDataFrame,
        name: tp.Optional[str] = None,
        copy: bool = True
) -> pd.Series:
    """Get the data of a `pyam.IamDataFram` as `pandas.Series` with MultiIndex.

    This function currently does the same as getting the private attribute
    `df._data` or `df._data.copy()` of the `pyam.IamDataFrame` directly, but
    should be used instead of that to avoid breaking changes in the future.

    Parameters
    ----------
    df : pyam.IamDataFrame
        IamDataFrame to get the data from.
    name : str, optional
        Name of the returned Series. Optional, defaults to None.
    copy : bool, optional
        Whether to return a copy of the data. If necessary, this parameter can
        be set to False to improve performance, but this carries some risks
        and may be removed or deprecated in the future. If False, the private
        attribute `df._data` is returned. Note that any changes made to the
        returned Series can then cause changes to and potentially corrupt the
        original IamDataFrame. Also, if the internal attributes of
        `pyam.IamDataFrame` are changed in the future, the `_data` attribute may
        be removed or changed, and any code using `copy=False` may then break.
        Optional, defaults to True.

    Returns
    -------
    pd.Series
        Data of the IamDataFrame as a Series.
    """
    data_ser: pd.Series = df._data if not copy else df._data.copy()
    if name is not None:
        data_ser.name = name
    return data_ser
###END def as_pandas_series


class MultipleCoordinateValuesError(ValueError):
    """Raised if an IamDataFrame has mulltiple coordinate values for given
    dimensions when only a single value is expected."""
    ...
###END class MultipleCoordinateValuesError


def broadcast_dims(
        df: pyam.IamDataFrame,
        target: pyam.IamDataFrame,
        dims: Sequence[str]
) -> pyam.IamDataFrame:
    """Make an IamDataFrame match coordinates of a target for given dimensions.

    The function takes an IamDataFrame with a single coordinate value for a list
    of dimensions, and renames the coordinates for those dimensions to match a
    target IamDataFrame, copying the data for each combination of the other
    dimensions.

    Note that the original IamDataFrame must have only a single coordinate value
    for each of the dimensions to match, or a `MultipleCoordinateValuesError`
    will be raised.

    Note the following for coordinate values in dimensions that are not
    broadcasted (not in `dims`), the returned IamDataFrame will have the same
    coordinate values as the original IamDataFrame `df`. This means that:
        - Any coordinate values in `df` that are not in `target` will be kept as
          they are, not dropped.
        - Any coordinate values in `target` that are not in `df` will not be
          added to the returned IamDataFrame.
    This is different from how, e.g., the `reindex` method of a pandas Series or
    DataFrame works, where the default behavior is to drop coordinate values not
    present in the target index, and add coordinate values in the target index
    that are not present in the original index, filling the values with NaN or
    another filler value.

    Parameters
    ----------
    df : pyam.IamDataFrame
        IamDataFrame to make match the target.
    target : pyam.IamDataFrame
        IamDataFrame to match the coordinates to.
    dims : sequence of str
        Dimensions to match the coordinates for.
    """
    for _dim in dims:
        if len(getattr(df, _dim)) > 1:
            raise MultipleCoordinateValuesError(
                f'The IamDataFrame has multiple coordinate values for the '
                f'dimension {_dim}. Can only broadcast on dimensions where '
                f'`df` only has a single coordinate value.'
            )
    broadcast_dim_original_values: dict[str, tp.Any] = {
        _dim: getattr(df, _dim)[0] for _dim in dims
    }
    df_broadcasted: pyam.IamDataFrame = df.copy()
    for _dim in dims:
        df_broadcasted_new_components: list[pyam.IamDataFrame] = [
            notnone(
                df_broadcasted.rename(
                    {_dim: {broadcast_dim_original_values[_dim]: _targetval}}
                )
            )
            for _targetval in getattr(target, _dim)
        ]
        df_broadcasted = pyam.concat(df_broadcasted_new_components)
    # df_broadcasted: pyam.IamDataFrame = pyam.concat(
    #     [
    #         df.rename({_dim: {broadcast_dim_original_values[_dim]: _targetval}})
    #         for _dim in dims for _targetval in getattr(target, _dim)
    #     ]
    # )
    return df_broadcasted
###END def broadcast_dims



