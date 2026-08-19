"""Base classes that extend pea.Criterion to entire IAM output timeseries.

The classes in this module are used to rate year-by-year differences between
a given IAM output timeseries and a reference timeseries. This replaces
previously started work that aimed to do the same through an entirely separate
class hierarchy (in the module `iam_vetter_core`). The new approach instead
leverages and integrates with the existing `Criterion` class and related
methods of the `pathways-ensemble-analysis` package (`pea`).
"""
import typing as tp
from collections.abc import Iterable, Callable, Iterator, Mapping
from enum import StrEnum
import dataclasses
import functools
import logging

import pyam
import pandas as pd
from pandas.core.indexes.frozen import FrozenList
from pandas.core.groupby import SeriesGroupBy
import pathways_ensemble_analysis as pea
from pathways_ensemble_analysis.criteria.base import Criterion

from .. import pyam_helpers
from .dims import (
    IamDimNames,
    DIM,
    UnknownDimensionNameError,
)



@dataclasses.dataclass(frozen=True)
class AggFuncTuple:
    """Class to hold an aggregation function and its parameter values.
    
    Fields
    ------
    func : Callable[[SeriesGroupBy], pandas.Series]
        The aggregation function to be applied. This field should not be set
        directly, instead use the `agg_func` parameter of the `__init__`
        method of this class.
    args : Iterable, optional
        The positional arguments to be passed to the aggregation function.
    kwargs : dict, optional
        The keyword arguments to be passed to the aggregation function.

    Init Parameters
    ---------------
    agg_func : Callable[[pandas.Series], float] or str
        The aggregation function to be applied by a pandas `SeriesGroupBy`
        object to aggregate over a given dimension, or the name of a method
        of the pandas `SeriesGroupBy` class.
    """

    class AggFunc(tp.Protocol):
        def __call__(self, s: pd.Series, /, *args, **kwargs) -> float:
            ...
    class GroupByAggMethod(tp.Protocol):
        def __call__(self, g: SeriesGroupBy, /, *args, **kwargs) -> pd.Series:
            ...
    ###END class AggFuncTuple.AggFunc

    func: AggFunc | str
    args: Iterable[tp.Any] = ()
    kwargs: dict[str, tp.Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.func, str):
            # Check that the attribute named `func` of `pandas.SeriesGroupBy`
            # is a method of `pandas.SeriesGroupBy`.
            groupby_attr: tp.Any = getattr(SeriesGroupBy, self.func)
            if not callable(getattr(SeriesGroupBy, self.func)):
                raise TypeError(
                    f'`{self.func}` is not a callable method of '
                    '`pandas.SeriesGroupBy`.'
                )
        elif not callable(self.func):
            raise TypeError('`func` must be a string or callable.')
    ###END def AggFuncTuple.__post_init__

    # Define iterator protocol to be able to use the class more like a tuple,
    # and `keys`, `values` and `__getitem__` to be able to use it like a
    # mapping.
    def __iter__(self) -> Iterator[tp.Any]:
        return iter(dataclasses.astuple(self))
    def keys(self) -> Iterable[str]:
        return dataclasses.asdict(self).keys()
    def values(self) -> Iterable[tp.Any]:
        return dataclasses.asdict(self).values()
    def __getitem__(self, key: str|int) -> tp.Any:
        if isinstance(key, int):
            return dataclasses.astuple(self)[key]
        elif isinstance(key, str):
            return dataclasses.asdict(self)[key]
    ###END def AggFuncTuple.__iter__

###END class AggFuncTuple


class AggDimOrder(StrEnum):
    """The order in which aggregations should be performed.
    
    The class defines which order to apply aggregations in when calling the
    `get_values` method of the `TimeseriesRefCriterion` class. That method needs
    to aggregate over both time and regions after calling `compare`.
    """
    TIME_FIRST = 'time_first'
    REGION_FIRST = 'region_first'
###END class AggDimOrder


class AggDims(StrEnum):
    """Which dimensions to aggregate over.

    The enum is used by the `TimeseriesRefCriterion.get_values` method parameter
    `agg_dims` to specify which dimensions to aggregate over, and may be used
    by other methods that similarly need to specify whether to aggregate over
    time, regions, or both.
    """
    TIME = 'time'
    REGION = 'region'
    TIME_AND_REGION = 'both'
    NO_AGGREGATION = 'none'
###END class AggDims


class TimeseriesRefCriterion(Criterion):
    """Base class for criteria that compare IAM output timeseries.

    This class is a subclass of `pea.Criterion` that is designed to compare
    year-by-year differences between a given IAM output timeseries and a
    reference timeseries. In addition to the method `get_value` that all
    `pea.Criterion` subclasses use to provide a single value for a given
    pathway, this class is designed to permit comparisons for all years and for
    multiple regions simultaneously, and therefore provides a method
    `compare` that can be used to get a fully disaggregated comparison (for each
    year and available region) or a partially aggregated one (aggregated over
    only time or only regions).

    The `get_value` method must nevertheless return a single value for a given
    pathway, i.e., a `pandas.Series` with only the levels `model` and
    `scenario` in the index. To achieve this, the user must pass functions that
    aggregate over regions and over time through the `region_agg` and `time_agg`
    parameters of the `__init__` method.

    Note that unlike the `pathways-ensemble-analysis.Criterion` base class,
    this class is intended to be able to check data for multiple regions at
    once, and the `__init__` method therefore does not take a `region`
    parameter. Please filter unwanted regions out of both the reference data
    before passing it to the `__init__` method, and from the data to be vetted
    before passing it to the `get_values` method. If you only intend to pass in
    reference data and data to be vetted for a single region, you can pass in
    `"first"` to the `region_agg` parameter of the `__init__` method to avoid
    having to construct an aggregation function (the data will then be
    "aggregated" using the `first` method of the `pandas.GroupBy` class, which
    simply returns the first value for each group, each of which should only
    contain a single value if the data only contains a single region).


    Init parameters
    ---------------
    criterion_name : str
        The name of the criterion.
    reference : pyam.IamDataFrame
        The reference timeseries to compare against. *NB!* The original passed
        object itself is stored and compared against, not a copy. This is done
        to conserve memory, and allow for defining multiple criteria with the
        same reference without taking up additional memory. Ensure that you do
        not unintentionally modify the reference object after passing it in.
    comparison_function : callable
        The function to use to compare the timeseries. The function should take
        two `pyam.IamDataFrame` objects as positional arguments and return a
        `pandas.Series` with comparison values (like differences, ratios or
        other difference measures). The first `IamDataFrame` should be one being
        compared to (`self.reference`) and the second one the object to be
        compared. The design of this base class implicitly assumes that the
        returned `Series` has the same index as the intersection of the indexes
        of the two `IamDataFrame`s (after broadcasting), but this is not
        enforced. You can also pass functions that compare the underlying
        `pandas.Series` objects, which can be significantly faster, by using the
        `pyam_series_comparison` decorator (in this module, see separate
        docstring). *NB!* The return value of `comparison_function` is specified
        as `pandas.Series` in order to allow more flexibility in dimensionality
        (i.e., index levels) than would be possible with a `pyam.IamDataFrame`.
        however, when possible, it is best practice to return a `pandas.Series`
        with a format that can be passed directly to `pyam.IamDataFrame` to
        construct a full `pyam.IamDataFrame` object (albeit with empty
        metadata). *NB!* It is the responsibility of the user to ensure that the
        returned `Series` has the correct units in the `unit` index level. If
        the `pyam_series_comparison` decorator is used, the units in the input
        will be made compatible before computation so that the resulting values
        are likely to be correct, but the unit name may no longer be correct.
        For example, if the comparison takes ratios or precentagewise
        differences, the units of the output will not be the same as the units
        of the inputs, and the user is responsible for ensuring that this is
        correctly reflected in the `unit` index level of the returned `Series`.
    region_agg : AggFuncTuple, tuple, callable or str
        The function to use to aggregate the timeseries over regions before
        calling `self.get_values`. If the function does not need to take any
        arguments, it should be either a callable that takes a `pandas.Series`
        and returns a float, or a string that is a method name of the pandas
        `SeriesGroupBy` class. If it takes arguments, it should be a 2- or 
        3-tuple of the form `(func, args, kwargs)`, where `func` is a callable
        or string, or an `AggFuncTuple` object (defined in this module).
    time_agg : AggFuncTuple, tuple, callable or str
        The function to use to aggregate the timeseries over time before
        calling `self.get_values`. Must fulfill the same requirements as
        `region_agg`.
    agg_dim_order: AggDimOrder or str, optional
        Which order to apply aggregations in when calling `self.get_values`.
        Should be an `AggDimOrder` enum, or a string that is equal to one of
        the enum values. Defaults to `AggDimOrder.REGION_FIRST`.
    default_agg_dims: AggDims or str, optional
        Which dimensions out of time and region to aggregate over by default
        when calling `self.get_values`. Should be an `AggDims` enum, or a string
        that is equal to one of the enum values. See the docstring of
        `self.get_values` for valid options. Defaults to `"both"`
        (i.e., `AggDims.TIME_AND_REGION`).
    broadcast_dims : iterable of str, optional
        The dimensions to broadcast over when comparing the timeseries. This
        should be a subset of the dimensions of the `reference` timeseries.
        `reference` should only have one value for each of these dimensions, or
        a `ValueError` will be raised. `reference` will be broadcast to the
        values of thsese dimensions in the `IamDataFrame` being comopared to
        before being passed to `comparison_function`. Optional, defaults to
        `('model', 'scenario')`.
    rating_function : callable, optional
        The function to use to rate the comparison values. This function should
        take and return single numbers. Optional, by default equals the identity
        function.
    dim_names : dim.IamDimNames, optional
        The dimension names of the reference `IamDataFrame`s used for reference
        and to be vetted. Optional, defaults to `dims.DIM`
    *args, **kwargs
        Additional arguments to be passed to the superclass `__init__` method.
        See the documentation of `pathways-ensemble-analysis.Criterion` for
        more information.

    Methods
    -------
    get_values(iamdf: pyam.IamDataFrame) -> pd.Series
        Returns the comparison values for the given `IamDataFrame`, after
        broadcasting and other processing and applying
        `self.comparison_function`. The values are returned as a
        `pandas.Series`, but in a form that can be converted directly to a
        `pyam.IamDataFrame` by passing it to the `pyam.IamDataFrame` __init__
        method. The returned `Series` from `TimeSeriesRefCriterion.get_values`
        will generally have the same index as the intersection of the indexes of
        the `IamDataFrame` and the `reference` timeseries (after broadcasting),
        but this is not enforced.
    rate(s: pd.Series) -> pd.Series
        Rates the comparison values in the given `pandas.Series` using
        `self.rating_function`. The returned `Series` will usually be an
        aggregate over years, and hence have an index without the 'year' level.
    """

    AggFuncArg: tp.TypeAlias = AggFuncTuple \
        | tuple[
            AggFuncTuple.AggFunc|str,
            Iterable[tp.Any],
            dict[str, tp.Any],
        ] \
        | AggFuncTuple.AggFunc \
        | str

    def __init__(
            self,
            criterion_name: str,
            reference: pyam.IamDataFrame,
            comparison_function: tp.Callable[
                [pyam.IamDataFrame, pyam.IamDataFrame], pd.Series
            ],
            region_agg: AggFuncArg,
            time_agg: AggFuncArg,
            agg_dim_order: AggDimOrder | str = AggDimOrder.REGION_FIRST,
            default_agg_dims: AggDims | str = AggDims.TIME_AND_REGION,
            broadcast_dims: Iterable[str] = ('model', 'scenario'),
            rating_function: Callable[[float], float] = lambda x: x,
            dim_names: IamDimNames = DIM,
            *args,
            **kwargs,
    ):
        self.reference: pyam.IamDataFrame = reference
        self.comparison_function: Callable[
            [pyam.IamDataFrame, pyam.IamDataFrame], pd.Series
        ] = comparison_function
        self._time_agg: AggFuncTuple = self._make_agg_func_tuple(time_agg)
        self._region_agg: AggFuncTuple = self._make_agg_func_tuple(region_agg)
        self.agg_dim_order: AggDimOrder = AggDimOrder(agg_dim_order)
        self.default_agg_dims: AggDims = AggDims(default_agg_dims)
        # Raise ValueError if `broadcast_dims` is not a subset of `reference.dimensions`
        if any(
                _dim not in reference.dimensions for _dim in broadcast_dims
        ):
            raise UnknownDimensionNameError('`broadcast_dims` must be a subset '
                                            'of `reference.dimensions`')
        self.dim_names: IamDimNames = dim_names
        self.broadcast_dims: list[str] = list(broadcast_dims)
        super().__init__(
            criterion_name=criterion_name,
            region='*',
            rating_function=rating_function,
            *args,
            **kwargs
        )
    ###END def TimeseriesRefCriterion.__init__

    def _make_agg_func_tuple(self, agg_func: AggFuncArg) -> AggFuncTuple:
        if isinstance(agg_func, AggFuncTuple):
            return agg_func
        if isinstance(agg_func, str):
            return AggFuncTuple(agg_func)
        if isinstance(agg_func, tuple):
            return AggFuncTuple(*agg_func)
        if callable(agg_func):
            return AggFuncTuple(agg_func)
        raise TypeError(f'`agg_func` must be a string, tuple, or callable.')
    ###END def _make_agg_func_tuple

    def _aggregate_time(self, s: pd.Series) -> pd.Series:
        """Aggregate Series returned by `self.compare` over time."""
        agg_func_tuple: AggFuncTuple = self._time_agg
        return s.groupby(
            tp.cast(FrozenList, s.index.names) \
                .difference([self.dim_names.TIME]),
        ).agg(
            agg_func_tuple.func,
            *agg_func_tuple.args,
            **agg_func_tuple.kwargs
        )
    ###END def TimeseriesRefCriterion._aggregate_time

    def _aggregate_region(self, s: pd.Series) -> pd.Series:
        """Aggregate Series returned by `self.compare` over regions."""
        agg_func_tuple: AggFuncTuple = self._region_agg
        return s.groupby(
            tp.cast(FrozenList, s.index.names) \
                .difference([self.dim_names.REGION]),
        ).agg(
            agg_func_tuple.func,
            *agg_func_tuple.args,
            **agg_func_tuple.kwargs
        )
    ###END def TimeseriesRefCriterion._aggregate_region

    def aggregate_time_and_region(self, s: pd.Series) -> pd.Series:
        """Aggregate Series returned by `self.compare` over time and regions,
        
        This method is used to aggregate the output from `self.compare` before
        passing it to `self.get_values`. Aggregation over time and regions is
        done in the order specified by the `agg_dim_order` parameter passed to
        the `__init__` method.
        """
        if self.agg_dim_order == AggDimOrder.REGION_FIRST:
            return self._aggregate_time(self._aggregate_region(s))
        if self.agg_dim_order == AggDimOrder.TIME_FIRST:
            return self._aggregate_region(self._aggregate_time(s))
        raise RuntimeError(f'Unknown `agg_dim_order` {self.agg_dim_order}.')
    ###END def TimeseriesRefCriterion.aggregate_time_and_region

    def compare(
            self,
            iamdf: pyam.IamDataFrame,
            filter: tp.Optional[Mapping[str, tp.Any]] = None,
            join: tp.Literal['inner', 'outer', 'reference', 'input', None] \
                = None,
    ) -> pd.Series:
        """Return comparison values for the given `IamDataFrame`.

        This method returns the comparison values for the given `IamDataFrame`,
        after broadcasting and other processing and applying
        `self.comparison_function`. The values are returned as a
        `pandas.Series`, but in a form that can be converted directly to a
        `pyam.IamDataFrame` by passing it to the `pyam.IamDataFrame` __init__
        method. The returned `Series` from `TimeSeriesRefCriterion.get_values`
        will generally have the same index as the intersection of the indexes of
        the `IamDataFrame` and the `reference` timeseries (after broadcasting),
        but this is not enforced.

        Parameters
        ----------
        iamdf : pyam.IamDataFrame
            The `IamDataFrame` to get comparison values for.
        filter : Mapping[str, tp.Any], optional
            Filter to apply to the reference data `self.reference` before
            performing the comparison. Should be a dict that can be expanded
            (`**filter`) and passed to `self.reference.filter`.
        join : `"inner"`, `"outer"`, `"reference"`, `"input"` or `None`
            Whether and how to join the reference data and the input `iamdf`
            before comparing. The operation acts similarly to a join or merge,
            and is applied after broadcasting and filtering (if `filter` is
            specified) the reference data, but before comparing. If `join` is
            specified (i.e., not `None`), the output will in most cases have the
            same index as that resulting from the join operation on the `model`,
            `scenario`, `region`, `variable` and `year` columns. The `unit`
            column is ignored, in order to avoid treating variables that have
            different units in `iamdf` and in the reference data as distinct.
            Note that this can cause problems if either data set has more than
            one unit for the same combination of model, scenario, region and
            variable, but this probably should not occur in practice unless
            there is something wrong or not standards-compliant with the data.
            The valid values are:
                - `"inner"`: Use the intersection of the indexes of the `iamdf`
                  and the `reference` timeseries.
                - `"outer"`: Use the union of the indexes of the `iamdf` and the
                  `reference` timeseries.
                - `"reference"`: Use the index of the `reference` timeseries
                - `"input"`: Use the index of the `iamdf` timeseries
                - `None`: Do not perform any join, just perform the comparison
                  operation directly. If `self.comparison_function` is a plain
                  arithmetic operator or other binary operator, the result will
                  in most cases be the same as for `"outer"`, except possibly
                  for ordering.
            In all cases when referring to the index of `reference`, the index
            of `self.reference` after broadcasting and filtering is meant. For
            `outer` and `inner`, the resulting index will usually be ordered in
            the same way as `iamdf`, though the internal sorting of
            `pyam.IamDataFrame` may change this.

        Returns
        -------
        pd.Series
            The comparison values for the given `IamDataFrame`.
        """
        reference: pyam.IamDataFrame
        if filter is not None:
            reference = self.reference.filter(**filter)  # pyright: ignore[reportAssignmentType]
        else:
            reference = self.reference
        ref = pyam_helpers.broadcast_dims(reference, iamdf, self.broadcast_dims)
        if join is not None:
            _ref_data: pd.Series = \
                pyam_helpers.as_pandas_series(ref, copy=False)
            _ref_data_df: pd.DataFrame = _ref_data.reset_index(DIM.UNIT)
            _iamdf_data: pd.Series = \
                pyam_helpers.as_pandas_series(iamdf, copy=False)
            _iamdf_data_df: pd.DataFrame = _iamdf_data.reset_index(DIM.UNIT)
            _join_index: pd.Index
            if join == 'inner':
                _join_index = \
                    _ref_data_df.index.intersection(_iamdf_data_df.index,
                                                    sort=False)
            elif join == 'outer':
                _join_index = \
                    _ref_data_df.index.union(_iamdf_data_df.index, sort=False)
            elif join == 'reference':
                _join_index = _ref_data_df.index
            elif join == 'input':
                _join_index = _iamdf_data_df.index
            else:
                raise ValueError(f'Unknown join value {join}.')
            ref = pyam.IamDataFrame(
                data=_ref_data_df.reindex(_join_index) \
                    .set_index(DIM.UNIT, append=True) \
                        .reorder_levels(_ref_data.index.names),
            )
            iamdf = pyam.IamDataFrame(
                data=_iamdf_data_df.reindex(_join_index) \
                    .set_index(DIM.UNIT, append=True) \
                        .reorder_levels(_iamdf_data.index.names),
            )
        return self.comparison_function(ref, iamdf)
    ###END def TimeseriesRefCriterion.get_values

    def get_values(
            self,
            file: pyam.IamDataFrame,
            agg_dims: tp.Optional[AggDims] = None
    ) -> pd.Series:
        """Return comparison values aggregated over region and time.

        Parameters
        ----------
        file : pyam.IamDataFrame
            The `IamDataFrame` to get comparison values for.
        agg_dims : AggDims, Optional
            Which dimensions to aggregate over. Use an `AggDims` enum value or
            a string value. Valid values are:
                - `"time"`/`AggDims.TIME`: Aggregate over time
                - `"region"`/`AggDims.REGION`: Aggregate over regions
                - `"both"`/`AggDims.TIME_AND_REGION`: Aggregate over time and regions
                - `"none"`/`AggDims.NONE`: Do not aggregate (NB! not `None`!)
            The default is `"both"`, unless a different default has been
            specified using the `default_agg_dims` parameter of the `__init__`
            method. This is also the behavior that is expected of the superclass
            `get_values` method. Using other options may therefore result in
            unexpected behavior if used with other packages.  The `"none"`
            option is essentially the same as calling `.compare`.

        Returns
        -------
        pd.Series
            The comparison values for the given `IamDataFrame`.
        """
        if agg_dims is None:
            agg_dims = self.default_agg_dims
        match agg_dims:
            case AggDims.TIME_AND_REGION:
                return self.aggregate_time_and_region(self.compare(file))
            case AggDims.TIME:
                return self._aggregate_time(self.compare(file))
            case AggDims.REGION:
                return self._aggregate_region(self.compare(file))
            case AggDims.NO_AGGREGATION:
                return self.compare(file)
            case _:
                raise ValueError(f'Unknown agg_dims value {agg_dims}.')
    ###END def TimeseriesRefCriterion.get_values

###END class TimeseriesRefCriterion


@tp.overload
def pyam_series_comparison(
        func: Callable[[pd.Series, pd.Series], pd.Series],
        *,
        match_units: bool = True
) -> Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]:
    ...
@tp.overload
def pyam_series_comparison(
        *,
        match_units: bool = True
) -> Callable[
        [Callable[[pd.Series, pd.Series], pd.Series]],
        Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]
]:
    ...
def pyam_series_comparison(
        func: tp.Optional[Callable[[pd.Series, pd.Series], pd.Series]] = None,
        *,
        match_units: bool = True
) -> Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series] | \
    Callable[
        [Callable[[pd.Series, pd.Series], pd.Series]],
        Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]
    ]:
    """Convert function comparing `Series` to one comparing `IamDataFrame`s.

    The function is designed to be used as a decorator. The decorated function
    must take two `pandas.Series` objects as positional arguments and return a
    `pandas.Series`. By default, the units of the first `Series` will be
    converted to the units of the second `Series` before the comparison is
    made, using the `pyam_helpers.match_units` function. If you are sure that
    the units are already consistent, you can pass `match_units=False` to the
    decorator (an optional keyword argument) to skip this step and improve
    performance.
    """
    def decorator(
            _func: Callable[[pd.Series, pd.Series], pd.Series]
    ) -> Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]:
        @functools.wraps(_func)
        def wrapper(iamdf1: pyam.IamDataFrame, iamdf2: pyam.IamDataFrame) \
                -> pd.Series:
            if match_units:
                iamdf1 = pyam_helpers.make_consistent_units(
                    df=iamdf1,
                    match_df=iamdf2
                )
            return _func(
                pyam_helpers.as_pandas_series(iamdf1),
                pyam_helpers.as_pandas_series(iamdf2)
            )
        return wrapper
    if func is None:
        return decorator
    return decorator(func)
###END def pyam_series_comparison


def get_diff_comparison(
        absolute: bool = False,
        match_units: tp.Optional[bool] = None,
) -> Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]:
    """Get comparison function for difference between reference and data 

    Parameters
    ----------
    absolute : bool, optional
        Whether to return the absolute difference rather than the signed
        difference. If False, negative numbers denote that the reference is
        greater than the data. Optional, by default False.
    match_units : bool or None, optional
        Whether to ensure that the units of the reference and data are
        consistent. If None, the default of the `pyam_series_comparison`
        function decorator will be used. Optional, by default None.

    Returns
    -------
    Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]
        The comparison function.
    """
    diff_func: Callable[[pd.Series, pd.Series], pd.Series] = \
        (lambda _ref, _data: _data - _ref) if not absolute \
            else (lambda _ref, _data: (_data - _ref).abs())
    if match_units is None:
        return pyam_series_comparison(diff_func)
    else:
        return pyam_series_comparison(diff_func, match_units=match_units)
###END def get_diff_comparison

def get_ratio_comparison(
        div_by_zero_value: tp.Optional[float] = None,
        zero_by_zero_value: tp.Optional[float] = None,
        match_units: tp.Optional[bool] = None,
) -> Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]:
    """Get comparison function for ratio between reference and data.

    The returned function divides the data by the reference, i.e., gives 1.0
    where they are equal, less than 1.0 where the reference is greater than the
    data, and greater than 1.0 where the reference is less than the data.
    Subtract 1 and multiply by 100 to get the percentage difference.

    The function allows the user to specify how division by zero should be
    handled, i.e., in the case that a reference data point is zero, both in the
    case that the input data point (the numerator) is itself zero and in the
    case that it is non-zero, through the parameters `div_by_zero_value` and
    `zero_by_zero_value`, respectively. If a number is specified for these
    parameters, that number will be used in all cases of the corresponding type
    of zero-division. If they are None, the function will use whichever value
    is returned by `pandas` when dividing one `Series` by another that contains
    zero values.

    Parameters
    ----------
    div_by_zero_value : float, optional
        Value to use when dividing a non-zero value by zero. Optional, by
        default None (i.e., use the number returned by `pandas` when dividing by
        a Series that contains zero values). NB! This parameter is only used
        when a *non*-zero value is divided by zero. For zero-by-zero divisions,
        use the `zero_by_zero_value` parameter.
    zero_by_zero_value : float, optional
        Value to use when dividing zero by zero. Optional, by default None.
    match_units : bool or None, optional
        Whether to ensure that the units of the reference and data are
        consistent. If None, the default of the `pyam_series_comparison`
        function decorator will be used. Optional, by default None.

    Returns
    -------
    Callable[[pyam.IamDataFrame, pyam.IamDataFrame], pd.Series]
        The comparison function.
    """
    def _division_func(_ref: pd.Series, _data: pd.Series) -> pd.Series:
        _div_series: pd.Series = _data / _ref
        if div_by_zero_value is not None:
            _div_series = _div_series.mask(
                (_ref == 0.0) & (_data != 0.0),
                other=div_by_zero_value,
            )
        if zero_by_zero_value is not None:
            _div_series = _div_series.mask(
                (_ref == 0.0) & (_data == 0.0),
                other=zero_by_zero_value,
            )
        return _div_series
    if match_units is None:
        return pyam_series_comparison(_division_func)
    else:
        return pyam_series_comparison(_division_func, match_units=match_units)
###END def get_ratio_comparison
