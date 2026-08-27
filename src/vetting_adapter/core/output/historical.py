"""Output classes for diagnostic (non-blocking) historical-data comparisons.

Unlike `CriterionTargetRangeOutput`/`MultiCriterionTargetRangeOutput` (which
report pass/fail against a target range) and
`TimeseriesRefComparisonAndTargetOutput` (also pass/fail, against a ratio
target range), the classes here are purely diagnostic: they show a checked
scenario's values for a single variable/region side by side with historical
reference values, plus the percent difference, so a user can visually judge
how closely the scenario follows history. There is no pass/fail result.

Coverage is naturally partial: a checked dataset will often not have every
variable/region combination present in the reference data. Rather than
raising, `HistoricalComparisonOutput.is_applicable` lets callers check this
up front (see also `MultiHistoricalComparisonOutput`, which maps each checked
name to an ordered list of exact/alias/fallback-region
`HistoricalComparisonOutput` candidates, used to build a "not applicable"
listing for the checked dataset).
"""
from collections.abc import Iterable, Mapping, Sequence
import typing as tp

import pandas as pd
import pyam

from ... import pyam_helpers
from ...pdhelpers import replace_level_values
from ..criteria import (
    TimeseriesRefCriterion,
    pyam_series_comparison,
    ratio_reference_criterion,
)
from ..dims import DIM
from .base import NoWriter, ResultOutput



class HistoricalComparisonOutput(
    ResultOutput[
        TimeseriesRefCriterion,
        pyam.IamDataFrame,
        dict[str, pd.DataFrame],
        NoWriter,
        None,
    ]
):
    """Diagnostic side-by-side comparison of one variable/region to history.

    `prepare_output` returns a dict with (by default) the keys
    `"checked_values"` and `"historical_values"` (both wide DataFrames with
    years as columns), plus `"pct_diff"` unless `include_pct_diff` is False.
    All three, when present, share the same (model, scenario, region,
    variable, unit) row index and the same set of year columns: *every* year
    the checked data has for this variable/region (not just years the
    reference data happens to cover), optionally capped by
    `future_buffer_years`. Years the reference data doesn't cover show as
    `NaN` in `"historical_values"`/`"pct_diff"`, not as a dropped column --
    this lets a user see, e.g., a scenario's near-term trend continuing a few
    years past the last year the reference data covers.

    Init Parameters
    ----------------
    reference : pyam.IamDataFrame
        Reference data for a single variable and region (i.e.
        `reference.variable` and `reference.region` should each have exactly
        one value).
    variable, region : str
        The variable and region `reference` is for. Used to check
        applicability and (for `fallback_from_region`) to relabel checked
        data before comparing.
    broadcast_dims : iterable of str, optional
        Passed to `TimeseriesRefCriterion`. Optional, defaults to
        `('model', 'scenario')`, i.e. the single reference timeseries is
        compared against every model/scenario in the checked data (checks
        are model-agnostic).
    include_pct_diff : bool, optional
        Whether to include a `"pct_diff"` entry in the output. Optional,
        defaults to True. Set to False when the checked data doesn't
        actually match `region` (see `fallback_from_region`), since a
        percent difference between different regions is not meaningful.
    fallback_from_region : str, optional
        If given, `prepare_output`/`is_applicable` will look for `region`'s
        data under this region name in the checked data instead of `region`
        itself, relabeling it to `region` before comparing. Used for
        reference regions that are broader than a region the checked
        dataset is likely to use (e.g. a historical value reported for
        `"EU27+UK"`, shown against a checked dataset's plain `"EU27"`/`
        European Union (R9)"` data as an approximation). Optional, defaults
        to None (compare against `region` directly).
    future_buffer_years : int, optional
        If given, checked-data years beyond `max(reference.year) +
        future_buffer_years` are dropped from the output (and don't count
        towards `is_applicable`). Lets a checked scenario's near-term trend
        past the reference data's own horizon be shown for context, without
        showing its entire multi-decade horizon (most of which would just be
        an all-`NaN` reference/pct-diff column). Optional, defaults to None
        (no cap -- every checked year for this variable/region is shown).
    source_name, source_url : str, optional
        Human-readable citation for `reference`'s data, e.g. `"Eurostat"`
        and a link to the specific dataset. Purely informational -- stored
        as attributes for a caller (e.g. a UI) to display; not used by this
        class itself. Optional, default None (no citation available).
    checked_values_key, historical_values_key, pct_diff_key : str, optional
        Keys to use in the dict returned by `prepare_output`. Optional.
    """

    def __init__(
            self,
            *,
            reference: pyam.IamDataFrame,
            variable: str,
            region: str,
            broadcast_dims: Iterable[str] = ('model', 'scenario'),
            include_pct_diff: bool = True,
            fallback_from_region: tp.Optional[str] = None,
            future_buffer_years: tp.Optional[int] = None,
            source_name: tp.Optional[str] = None,
            source_url: tp.Optional[str] = None,
            checked_values_key: str = 'checked_values',
            historical_values_key: str = 'historical_values',
            pct_diff_key: str = 'pct_diff',
    ):
        self.reference: pyam.IamDataFrame = reference
        self.variable: str = variable
        self.region: str = region
        self.broadcast_dims: list[str] = list(broadcast_dims)
        self.include_pct_diff: bool = include_pct_diff
        self.fallback_from_region: tp.Optional[str] = fallback_from_region
        self.future_buffer_years: tp.Optional[int] = future_buffer_years
        self.source_name: tp.Optional[str] = source_name
        self.source_url: tp.Optional[str] = source_url
        self.checked_values_key: str = checked_values_key
        self.historical_values_key: str = historical_values_key
        self.pct_diff_key: str = pct_diff_key

        self._criterion_checked: TimeseriesRefCriterion = TimeseriesRefCriterion(
            criterion_name=f'{variable} ({region}) - checked values',
            reference=reference,
            comparison_function=pyam_series_comparison(match_units=False)(
                lambda _ref_s, _data_s: _data_s
            ),
            region_agg='first',
            time_agg='first',
            broadcast_dims=self.broadcast_dims,
            restrict_to_reference=True,
        )
        self._criterion_historical: TimeseriesRefCriterion = TimeseriesRefCriterion(
            criterion_name=f'{variable} ({region}) - historical reference',
            reference=reference,
            comparison_function=pyam_series_comparison(match_units=True)(
                lambda _ref_s, _data_s: _ref_s
            ),
            region_agg='first',
            time_agg='first',
            broadcast_dims=self.broadcast_dims,
            restrict_to_reference=True,
        )
        self._criterion_ratio: tp.Optional[TimeseriesRefCriterion] = \
            ratio_reference_criterion(
                criterion_name=f'{variable} ({region}) - ratio',
                reference=reference,
                broadcast_dims=self.broadcast_dims,
            ) if include_pct_diff else None
        super().__init__(criteria=self._criterion_checked, writer=NoWriter())
    ###END def HistoricalComparisonOutput.__init__

    def _relabel_to_reference_region(
            self,
            data: pyam.IamDataFrame,
    ) -> pyam.IamDataFrame:
        """Filter `data` to `self.fallback_from_region` and relabel it to
        `self.region`, so it can be compared to `self.reference` directly.
        Only used when `self.fallback_from_region` is not None.
        """
        filtered: pyam.IamDataFrame = data.filter(
            region=self.fallback_from_region,
        )  # pyright: ignore[reportAssignmentType]
        if filtered.empty:
            return filtered
        relabeled: pd.Series = replace_level_values(
            pyam_helpers.as_pandas_series(filtered),
            mapping={self.fallback_from_region: self.region},
            level_name='region',
        )
        return pyam.IamDataFrame(relabeled)
    ###END def HistoricalComparisonOutput._relabel_to_reference_region

    def _year_cutoff(self) -> tp.Optional[int]:
        """Last year to show, or None if `future_buffer_years` is not set."""
        if self.future_buffer_years is None:
            return None
        return max(self.reference.year) + self.future_buffer_years
    ###END def HistoricalComparisonOutput._year_cutoff

    def is_applicable(self, data: pyam.IamDataFrame) -> bool:
        """Whether `data` has any displayable values for this variable/region.

        Checks cheaply (without running the full comparison, which would
        raise if there is no overlap at all) whether `data` has any values
        for `self.variable`, for the region checked (`self.region`, or
        `self.fallback_from_region` if set), for at least one year that
        would actually be shown (i.e. not later than the
        `future_buffer_years` cutoff, if set). Unlike an earlier version of
        this method, this does *not* require the year to also be one that
        `self.reference` covers -- `prepare_output` shows every checked year
        up to the cutoff regardless, with `NaN` reference/pct-diff values
        for years the reference data doesn't cover.
        """
        check_region: str = self.fallback_from_region \
            if self.fallback_from_region is not None else self.region
        filtered: pyam.IamDataFrame = data.filter(
            region=check_region,
            variable=self.variable,
        )  # pyright: ignore[reportAssignmentType]
        if filtered.empty:
            return False
        cutoff: tp.Optional[int] = self._year_cutoff()
        if cutoff is None:
            return True
        return any(_year <= cutoff for _year in filtered.year)
    ###END def HistoricalComparisonOutput.is_applicable

    def _prepare_table(
            self,
            criterion: TimeseriesRefCriterion,
            data: pyam.IamDataFrame,
            *,
            index: tp.Optional[pd.Index] = None,
            columns: tp.Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        """Compare `criterion` against every year in `data` (not just years
        `self.reference` covers), then apply the `future_buffer_years` cutoff
        to the resulting wide (year-columns) DataFrame, if set.

        If `index`/`columns` are given, the result is reindexed to them
        (filling any missing rows/columns with `NaN`) as a final step. This
        is needed because `pyam.IamDataFrame` construction -- which
        `criterion.compare` does internally as part of the join -- silently
        drops rows with a `NaN` value; passing `checked_values`' own index
        and (post-cutoff) columns here restores those as explicit `NaN`
        cells for `historical_values`/`pct_diff`, rather than as dropped
        rows/columns, so all three tables always share the same shape.
        """
        series: pd.Series = criterion.compare(data, join='input')
        table: pd.DataFrame = series.unstack(level=DIM.TIME)
        cutoff: tp.Optional[int] = self._year_cutoff()
        if cutoff is not None:
            table = table[[_year for _year in table.columns if _year <= cutoff]]
        if index is not None or columns is not None:
            table = table.reindex(
                index=index if index is not None else table.index,
                columns=columns if columns is not None else table.columns,
            )
        return table
    ###END def HistoricalComparisonOutput._prepare_table

    def prepare_output(
            self,
            data: pyam.IamDataFrame,
            /,
            criteria: tp.Optional[TimeseriesRefCriterion] = None,
    ) -> dict[str, pd.DataFrame]:
        """Prepare the checked-vs-historical (and, usually, pct-diff) tables.

        Returns an empty DataFrame for each key if `self.is_applicable(data)`
        is False, rather than raising.
        """
        if not self.is_applicable(data):
            empty_df: pd.DataFrame = pd.DataFrame()
            output: dict[str, pd.DataFrame] = {
                self.checked_values_key: empty_df,
                self.historical_values_key: empty_df,
            }
            if self.include_pct_diff:
                output[self.pct_diff_key] = empty_df
            return output
        if self.fallback_from_region is not None:
            data = self._relabel_to_reference_region(data)
        checked_table: pd.DataFrame = \
            self._prepare_table(self._criterion_checked, data)
        output = {
            self.checked_values_key: checked_table,
            self.historical_values_key: self._prepare_table(
                self._criterion_historical, data,
                index=checked_table.index, columns=checked_table.columns,
            ),
        }
        if self.include_pct_diff and self._criterion_ratio is not None:
            ratio_table: pd.DataFrame = self._prepare_table(
                self._criterion_ratio, data,
                index=checked_table.index, columns=checked_table.columns,
            )
            output[self.pct_diff_key] = (ratio_table - 1.0) * 100.0
        return output
    ###END def HistoricalComparisonOutput.prepare_output

###END class HistoricalComparisonOutput


class MultiHistoricalComparisonOutput:
    """Registry of `HistoricalComparisonOutput` candidate lists for a checkset.

    Each named check maps to an ordered list of `HistoricalComparisonOutput`
    candidates to try in turn: the first (built by `vetting_adapter.core.spec`)
    always compares against the reference region directly
    (`fallback_from_region=None`, `include_pct_diff=True`); it may be followed
    by region-alias candidates (an exact match under a different region name,
    e.g. `"EU27"` for `"European Union (R9)"` -- still `include_pct_diff=True`,
    since it is the same region), and/or region-fallback candidates (a
    genuinely broader/narrower region used as an approximation --
    `include_pct_diff=False`, since the regions don't actually match). See
    `HistoricalComparisonOutput` and the `region_aliases`/`region_fallbacks`
    checkset spec fields in `vetting_adapter.core.spec`.

    This is intentionally not a `ResultOutput` subclass: unlike
    `MultiCriterionTargetRangeOutput`, this class is meant to be driven
    interactively (e.g. from a UI) by first checking applicability of each
    entry, not to produce one flat multi-sheet output in a single call.
    """

    def __init__(
            self,
            criteria: Mapping[str, Sequence[HistoricalComparisonOutput]],
    ):
        self.criteria: dict[str, list[HistoricalComparisonOutput]] = {
            _name: list(_candidates)
            for _name, _candidates in criteria.items()
        }
    ###END def MultiHistoricalComparisonOutput.__init__

    def get_applicable_output(
            self,
            name: str,
            data: pyam.IamDataFrame,
    ) -> tp.Optional[HistoricalComparisonOutput]:
        """Get the first applicable `HistoricalComparisonOutput` for `name`.

        Tries each candidate for `name` in order (see class docstring) and
        returns the first one that `.is_applicable(data)`. Callers can check
        the returned object's `.include_pct_diff` attribute to tell an exact
        or region-alias match (True) from a region-approximated one (False).
        Returns `None` if no candidate is applicable to `data`.
        """
        for _candidate in self.criteria[name]:
            if _candidate.is_applicable(data):
                return _candidate
        return None
    ###END def MultiHistoricalComparisonOutput.get_applicable_output

###END class MultiHistoricalComparisonOutput
