"""Build vetting check output objects from declarative YAML specs.

This module is the counterpart to `vetting_adapter.general_checks`: instead of
a fixed, hand-written check, it builds a check output object at runtime from a
small YAML spec, which is expected to live in a project's own definitions
repository (see `vetting_adapter.profiles`). This lets a project add its own
checks without writing any Python code or requiring a release of this
package.

Three spec `type`s are currently supported:

`reference_comparison`
    Checks that compare a scenario's timeseries to a reference timeseries by
    ratio (e.g., harmonization checks), backed by a reference data file. See
    `_build_reference_comparison` for the YAML schema.
`target_range`
    Checks with fixed target/range criteria for one or more single
    variable/year or change-over-time values, not backed by any reference
    data file. This mirrors what `vetting_adapter.general_checks.ar6_vetting`
    does in hand-written Python for the (project-agnostic) IPCC AR6 vetting
    criteria; use that module as a reference for what this spec type can
    express. See `_build_target_range` for the YAML schema.
`historical_comparison`
    Diagnostic (non-blocking) side-by-side comparison of a scenario's
    historical years to reference values, backed by a reference data file,
    for whichever variable/region/year combinations the checked dataset
    actually has. Unlike the other two types, there is no pass/fail result.
    See `_build_historical_comparison` for the YAML schema.

See `build_check_from_spec` for the entry point.
"""
from collections.abc import Callable
from pathlib import Path
import typing as tp

import numpy as np
import pyam
import yaml

from pathways_ensemble_analysis.criteria.base import (
    Criterion,
    SingleVariableCriterion,
    ChangeOverTimeCriterion,
)

from .criteria import ratio_reference_criterion
from .target_range import CriterionTargetRange, RatioTargetRange, RelativeRange
from .output.base import (
    CriterionTargetRangeOutput,
    MultiCriterionTargetRangeOutput,
    NoWriter,
)
from .output.column_names import CTCol
from .output.timeseries import (
    TimeseriesRefComparisonAndTargetOutput,
    TimeseriesRefFullComparisonOutput,
)
from .output.historical import (
    HistoricalComparisonOutput,
    MultiHistoricalComparisonOutput,
)



class CheckSpecError(ValueError):
    """Raised when a checkset YAML spec is missing or invalid."""
    ...
###END class CheckSpecError


SUPPORTED_CHECK_TYPES: tp.Final[tuple[str, ...]] = \
    ('reference_comparison', 'target_range', 'historical_comparison')

_CRITERION_TYPES: tp.Final[dict[str, tp.Type[Criterion]]] = {
    'single_variable': SingleVariableCriterion,
    'change_over_time': ChangeOverTimeCriterion,
}
"""Named `Criterion` subclasses that can be referenced by a `target_range`
checkset spec's `criteria[].criterion_type` field."""

_DISTANCE_FUNC_TYPES: tp.Final[tuple[str, ...]] = ('fixed_denominator',)
"""Named `distance_func` builders that can be referenced by a `target_range`
checkset spec's `criteria[].distance_func.type` field. See
`_build_distance_func`."""

_RATING_FUNCTIONS: tp.Final[dict[str|None, Callable[[float], float]]] = {
    None: lambda x: x,
    'identity': lambda x: x,
    'log_rms': lambda x: np.sqrt(np.log10(x)**2),
}
"""Named rating functions that can be referenced from a checkset spec by name,
rather than requiring the spec to embed Python code."""


def load_spec(spec_file: Path) -> dict:
    """Load and minimally validate a checkset YAML spec file."""
    with spec_file.open('r', encoding='utf-8') as _f:
        spec = yaml.safe_load(_f)
    if not isinstance(spec, dict):
        raise CheckSpecError(
            f'Checkset spec file "{spec_file}" must contain a YAML mapping.'
        )
    if 'name' not in spec:
        raise CheckSpecError(
            f'Checkset spec file "{spec_file}" is missing a "name" field.'
        )
    return spec
###END def load_spec


def build_check_from_spec(
        spec_file: Path,
        repo_root: Path,
) -> tuple[
    str,
    TimeseriesRefComparisonAndTargetOutput
        |MultiCriterionTargetRangeOutput
        |MultiHistoricalComparisonOutput,
]:
    """Build a check output object from a checkset YAML spec file.

    Parameters
    ----------
    spec_file : pathlib.Path
        Path to the checkset YAML spec file.
    repo_root : pathlib.Path
        Root directory that paths given in the spec (such as the reference
        data file) are resolved relative to. This is normally the root of the
        cloned definitions repository that `spec_file` was found in. Used by
        the `reference_comparison` and `historical_comparison` spec types.

    Returns
    -------
    (name, output) : tuple[str, TimeseriesRefComparisonAndTargetOutput|MultiCriterionTargetRangeOutput|MultiHistoricalComparisonOutput]
        The check name (from the spec's `name` field, used as the key in the
        check registry), and the constructed output object.

    Raises
    ------
    CheckSpecError
        If the spec is missing required fields, refers to an unsupported
        `type`/`comparison.method`/`criterion_type`, or refers to a
        reference data file that does not exist.
    """
    spec: dict = load_spec(spec_file)
    check_type: str = spec.get('type', 'reference_comparison')
    if check_type not in SUPPORTED_CHECK_TYPES:
        raise CheckSpecError(
            f'Unsupported checkset type "{check_type}" in "{spec_file}". '
            f'Supported types: {", ".join(SUPPORTED_CHECK_TYPES)}.'
        )
    output: TimeseriesRefComparisonAndTargetOutput \
        |MultiCriterionTargetRangeOutput|MultiHistoricalComparisonOutput
    if check_type == 'reference_comparison':
        output = _build_reference_comparison(spec, repo_root, spec_file=spec_file)
    elif check_type == 'target_range':
        output = _build_target_range(spec, spec_file=spec_file)
    else:  # check_type == 'historical_comparison'
        output = _build_historical_comparison(spec, repo_root, spec_file=spec_file)
    return spec['name'], output
###END def build_check_from_spec


def _build_reference_comparison(
        spec: dict,
        repo_root: Path,
        *,
        spec_file: Path,
) -> TimeseriesRefComparisonAndTargetOutput:
    """Build a `TimeseriesRefComparisonAndTargetOutput` from a spec of type
    `reference_comparison`. See `build_check_from_spec` for the YAML schema.
    """
    reference_cfg: dict = spec.get('reference', {})
    if 'file' not in reference_cfg:
        raise CheckSpecError(
            f'Checkset spec "{spec_file}" is missing "reference.file".'
        )
    reference_file: Path = repo_root / reference_cfg['file']
    if not reference_file.is_file():
        raise CheckSpecError(
            f'Reference data file "{reference_file}" (from checkset spec '
            f'"{spec_file}") does not exist.'
        )
    reference: pyam.IamDataFrame = pyam.IamDataFrame(reference_file)

    comparison_cfg: dict = spec.get('comparison', {})
    method: str = comparison_cfg.get('method', 'ratio')
    if method != 'ratio':
        raise CheckSpecError(
            f'Unsupported comparison method "{method}" in checkset spec '
            f'"{spec_file}" (only "ratio" is currently supported).'
        )
    target: float = comparison_cfg.get('target', 1.0)
    tolerance: tp.Optional[float] = comparison_cfg.get('tolerance')
    explicit_range: tp.Optional[list] = comparison_cfg.get('range')
    if explicit_range is None and tolerance is None:
        raise CheckSpecError(
            f'Checkset spec "{spec_file}" must specify either '
            '"comparison.tolerance" or "comparison.range".'
        )
    target_range: tuple[float, float]|RelativeRange = \
        tuple(explicit_range) if explicit_range is not None \
        else RelativeRange(1.0-tolerance, 1.0+tolerance)

    rating_key: str|None = spec.get('rating_function')
    if rating_key not in _RATING_FUNCTIONS:
        raise CheckSpecError(
            f'Unknown rating_function "{rating_key}" in checkset spec '
            f'"{spec_file}". Known values: '
            f'{", ".join(_k for _k in _RATING_FUNCTIONS if _k is not None)}.'
        )

    aggregation_cfg: dict = spec.get('aggregation', {})
    criterion = ratio_reference_criterion(
        criterion_name=spec['name'],
        reference=reference,
        region_agg=aggregation_cfg.get('region', 'max'),
        time_agg=aggregation_cfg.get('time', 'max'),
        broadcast_dims=aggregation_cfg.get(
            'broadcast_dims', ['model', 'scenario']
        ),
        rating_function=_RATING_FUNCTIONS[rating_key],
    )

    output_cfg: dict = spec.get('output', {})
    summary_key: str = output_cfg.get('summary_key', 'Summary')
    full_comparison_key: str = \
        output_cfg.get('full_comparison_key', 'Full comparison')

    def _make_summary_output(
            _target_range: RatioTargetRange,
    ) -> CriterionTargetRangeOutput:
        return CriterionTargetRangeOutput(
            criteria=_target_range,
            writer=NoWriter(),
        )
    ###END def _make_summary_output

    return TimeseriesRefComparisonAndTargetOutput(
        criteria=criterion,
        target_range_type=RatioTargetRange,
        target=target,
        range=target_range,
        timeseries_output_type=TimeseriesRefFullComparisonOutput,
        summary_output=_make_summary_output,
        full_comparison_key=full_comparison_key,
        summary_key=summary_key,
        writer=NoWriter(),
    )
###END def _build_reference_comparison


def _build_distance_func(
        entry: tp.Optional[dict],
        *,
        criterion_name: str,
        spec_file: Path,
) -> tp.Optional[Callable[[float], float]]:
    """Build a `distance_func` callable from a `criteria[].distance_func` entry.

    Returns None if `entry` is None, meaning `CriterionTargetRange` should use
    its own default distance function.

    Only one type is currently supported: `fixed_denominator`, which builds
    `lambda x: x / value`. This is meant for criteria whose target is 0 (so
    the default distance function would divide by zero for values below the
    target), mirroring the `distance_func` used for the "CCS from energy
    2020" criterion in `vetting_adapter.general_checks.ar6_vetting`.
    """
    if entry is None:
        return None
    if not isinstance(entry, dict) or 'type' not in entry:
        raise CheckSpecError(
            f'"distance_func" for criterion "{criterion_name}" in checkset '
            f'spec "{spec_file}" must be a mapping with a "type" field.'
        )
    func_type: str = entry['type']
    if func_type not in _DISTANCE_FUNC_TYPES:
        raise CheckSpecError(
            f'Unsupported distance_func type "{func_type}" for criterion '
            f'"{criterion_name}" in checkset spec "{spec_file}". Supported '
            f'types: {", ".join(_DISTANCE_FUNC_TYPES)}.'
        )
    if 'value' not in entry:
        raise CheckSpecError(
            f'distance_func of type "{func_type}" for criterion '
            f'"{criterion_name}" in checkset spec "{spec_file}" is missing '
            '"value".'
        )
    denominator: float = float(entry['value'])
    return lambda x: x / denominator
###END def _build_distance_func


def _build_criterion_target_range(
        entry: dict,
        *,
        spec_file: Path,
) -> CriterionTargetRange:
    """Build a single `CriterionTargetRange` from a `criteria[]` list entry.

    See `_build_target_range` for the YAML schema of `entry`.
    """
    if not isinstance(entry, dict) or 'name' not in entry:
        raise CheckSpecError(
            f'Each item in "criteria" in checkset spec "{spec_file}" must be '
            'a mapping with a "name" field.'
        )
    name: str = entry['name']
    criterion_type: str = entry.get('criterion_type', 'single_variable')
    if criterion_type not in _CRITERION_TYPES:
        raise CheckSpecError(
            f'Unknown criterion_type "{criterion_type}" for criterion '
            f'"{name}" in checkset spec "{spec_file}". Supported types: '
            f'{", ".join(_CRITERION_TYPES)}.'
        )
    for required in ('region', 'year', 'variable', 'target'):
        if required not in entry:
            raise CheckSpecError(
                f'Criterion "{name}" in checkset spec "{spec_file}" is '
                f'missing "{required}".'
            )
    unit: tp.Optional[str] = entry.get('unit')
    criterion_kwargs: dict[str, tp.Any] = dict(
        criterion_name=name,
        region=entry['region'],
        year=int(entry['year']),
        variable=entry['variable'],
    )
    criterion: Criterion
    if criterion_type == 'single_variable':
        criterion = SingleVariableCriterion(unit=unit, **criterion_kwargs)
    else:  # criterion_type == 'change_over_time'
        if 'reference_year' not in entry:
            raise CheckSpecError(
                f'Criterion "{name}" of type "change_over_time" in checkset '
                f'spec "{spec_file}" is missing "reference_year".'
            )
        criterion = ChangeOverTimeCriterion(
            reference_year=int(entry['reference_year']),
            **criterion_kwargs,
        )

    range_keys: list[str] = \
        [_k for _k in ('range', 'relative_range', 'tolerance') if _k in entry]
    if len(range_keys) > 1:
        raise CheckSpecError(
            f'Criterion "{name}" in checkset spec "{spec_file}" specifies '
            f'more than one of "range", "relative_range", "tolerance" '
            f'({", ".join(range_keys)}); only one may be given.'
        )
    target_range: tp.Optional[tuple[float, float]|RelativeRange]
    if 'range' in entry:
        target_range = tuple(float(_v) for _v in entry['range'])  # pyright: ignore[reportAssignmentType]
    elif 'relative_range' in entry:
        target_range = RelativeRange(*entry['relative_range'])
    elif 'tolerance' in entry:
        tolerance: float = float(entry['tolerance'])
        target_range = RelativeRange(1.0-tolerance, 1.0+tolerance)
    else:
        target_range = None

    distance_func: tp.Optional[Callable[[float], float]] = _build_distance_func(
        entry.get('distance_func'), criterion_name=name, spec_file=spec_file,
    )

    return CriterionTargetRange(
        criterion=criterion,
        target=float(entry['target']),
        range=target_range,
        unit=unit,
        distance_func=distance_func,
    )
###END def _build_criterion_target_range


def _parse_ctcol(value: str, *, spec_file: Path) -> CTCol:
    try:
        return CTCol(value)
    except ValueError:
        raise CheckSpecError(
            f'Unknown output column "{value}" in checkset spec "{spec_file}". '
            f'Supported columns: {", ".join(_c.value for _c in CTCol)}.'
        ) from None
###END def _parse_ctcol


def _build_target_range(
        spec: dict,
        *,
        spec_file: Path,
) -> MultiCriterionTargetRangeOutput:
    """Build a `MultiCriterionTargetRangeOutput` from a spec of type
    `target_range`.

    This spec type is for checks with one or more fixed target/range
    criteria, each evaluated against a single variable/region/year (or a
    change over time between two years), not backed by any reference data
    file. It mirrors what `vetting_adapter.general_checks.ar6_vetting` does
    in hand-written Python for the built-in IPCC AR6 vetting criteria; use
    that module as a reference for the kinds of criteria this spec type can
    express.

    YAML schema
    -----------
    name : str
        Name of the checkset (used as the key in the check registry).
    type : str
        Must be `"target_range"`.
    criteria : list[dict]
        List of criterion specs, each a mapping with the following keys:

        name : str
            Name of the criterion (used as its key in the output, and as the
            `criterion_name` passed to the underlying `Criterion` class).
        criterion_type : str, optional
            Either `"single_variable"` (default) or `"change_over_time"`.
        region, year, variable : required
            Passed to the underlying `Criterion` class.
        reference_year : required if `criterion_type` is `"change_over_time"`
            Passed to `ChangeOverTimeCriterion`.
        unit : str, optional
            Unit of `target` and of the criterion's values.
        target : float, required
            Target value for the criterion.
        range, relative_range, tolerance : optional, mutually exclusive
            `range` is an explicit `[lower, upper]` pair of absolute values.
            `relative_range` is a `[lower, upper]` pair of multipliers of
            `target` (built into a `RelativeRange`). `tolerance` is a single
            symmetric relative tolerance around `target` (equivalent to
            `relative_range: [1-tolerance, 1+tolerance]`). If none is given,
            the criterion has no range (only a target).
        distance_func : dict, optional
            Override for the default distance function; see
            `_build_distance_func`.
    output : dict, optional
        columns : list[str], optional
            Subset/order of `"in_range"`, `"distance"`, `"value"` to include
            in each criterion's output. Defaults to all three.
        column_titles : dict[str, str], optional
            Mapping from column name (as in `columns`) to display title.
        summary_keys : dict[str, str], optional
            Mapping from column name to the key to use for a summary table
            (across all criteria) of that column. If given, summary tables
            are included in the output in addition to the per-criterion
            tables.
    """
    criteria_specs: tp.Optional[list] = spec.get('criteria')
    if not criteria_specs:
        raise CheckSpecError(
            f'Checkset spec "{spec_file}" of type "target_range" must have a '
            'non-empty "criteria" list.'
        )
    criteria: dict[str, CriterionTargetRange] = {}
    for _entry in criteria_specs:
        _target_range: CriterionTargetRange = \
            _build_criterion_target_range(_entry, spec_file=spec_file)
        if _target_range.name in criteria:
            raise CheckSpecError(
                f'Duplicate criterion name "{_target_range.name}" in '
                f'checkset spec "{spec_file}".'
            )
        criteria[_target_range.name] = _target_range

    output_cfg: dict = spec.get('output', {})
    columns: tp.Optional[list[CTCol]] = [
        _parse_ctcol(_c, spec_file=spec_file) for _c in output_cfg['columns']
    ] if 'columns' in output_cfg else None
    column_titles: tp.Optional[dict[CTCol, str]] = {
        _parse_ctcol(_k, spec_file=spec_file): _v
        for _k, _v in output_cfg['column_titles'].items()
    } if 'column_titles' in output_cfg else None
    summary_keys: tp.Optional[dict[CTCol, str]] = {
        _parse_ctcol(_k, spec_file=spec_file): _v
        for _k, _v in output_cfg['summary_keys'].items()
    } if 'summary_keys' in output_cfg else None

    return MultiCriterionTargetRangeOutput(
        criteria=criteria,
        writer=NoWriter(),
        columns=columns,
        column_titles=column_titles,
        summary_keys=summary_keys,
    )
###END def _build_target_range


def _build_historical_comparison(
        spec: dict,
        repo_root: Path,
        *,
        spec_file: Path,
) -> MultiHistoricalComparisonOutput:
    """Build a `MultiHistoricalComparisonOutput` from a spec of type
    `historical_comparison`.

    This spec type is for diagnostic (non-blocking) side-by-side comparisons
    of a checked scenario's values to historical reference values, for
    whichever variable/region/year combinations the checked dataset actually
    has. One named check is built automatically for each unique
    (variable, region) pair found in the reference data file -- there is no
    need to enumerate them in the YAML spec.

    YAML schema
    -----------
    name : str
        Name of the checkset (used as the key in the check registry).
    type : str
        Must be `"historical_comparison"`.
    reference : dict
        file : str, required
            Path (relative to `repo_root`) to the reference data file (in
            any format `pyam.IamDataFrame` can read, e.g. `.xlsx` or `.csv`).
    region_aliases : dict[str, list[str]], optional
        Maps a reference region name to a list of alternative names that
        refer to the *same* region (e.g. a plainer or more common spelling
        that a checked dataset might use instead of the canonical name).
        Tried, in order, if the reference region itself is not present in
        the checked dataset. Since these are the same region, percent
        difference is still computed when an alias matches.
    region_fallbacks : dict[str, str], optional
        Maps a reference region name to a fallback region name to look for
        in the checked dataset if neither the reference region nor any of
        its `region_aliases` are present. Unlike `region_aliases`, this is
        for reference regions that are genuinely broader/narrower than a
        region the checked dataset is likely to use (e.g. a historical
        value reported for a wider country aggregate than the checked
        dataset's own region): when the fallback is used, no percent
        difference is computed (only the checked and historical values,
        since the regions do not actually match).
    comparison : dict, optional
        broadcast_dims : list[str], optional
            Passed to `HistoricalComparisonOutput`. Optional, defaults to
            `["model", "scenario"]`, i.e. checks are model-agnostic.
    """
    reference_cfg: dict = spec.get('reference', {})
    if 'file' not in reference_cfg:
        raise CheckSpecError(
            f'Checkset spec "{spec_file}" is missing "reference.file".'
        )
    reference_file: Path = repo_root / reference_cfg['file']
    if not reference_file.is_file():
        raise CheckSpecError(
            f'Reference data file "{reference_file}" (from checkset spec '
            f'"{spec_file}") does not exist.'
        )
    reference: pyam.IamDataFrame = pyam.IamDataFrame(reference_file)

    region_aliases: dict[str, list[str]] = spec.get('region_aliases', {}) or {}
    if not isinstance(region_aliases, dict) or not all(
            isinstance(_v, list) for _v in region_aliases.values()
    ):
        raise CheckSpecError(
            f'"region_aliases" in checkset spec "{spec_file}" must be a '
            'mapping from reference region name to a list of alias names.'
        )

    region_fallbacks: dict[str, str] = spec.get('region_fallbacks', {}) or {}
    if not isinstance(region_fallbacks, dict):
        raise CheckSpecError(
            f'"region_fallbacks" in checkset spec "{spec_file}" must be a '
            'mapping from reference region name to fallback region name.'
        )

    comparison_cfg: dict = spec.get('comparison', {})
    broadcast_dims: list[str] = \
        list(comparison_cfg.get('broadcast_dims', ['model', 'scenario']))

    criteria: dict[str, list[HistoricalComparisonOutput]] = {}
    variable_region_pairs: set[tuple[str, str]] = set(
        reference.data[['variable', 'region']]
            .drop_duplicates()
            .itertuples(index=False, name=None)
    )
    for _variable, _region in sorted(variable_region_pairs):
        _name: str = f'{_variable} ({_region})'
        if _name in criteria:
            raise CheckSpecError(
                f'Duplicate (variable, region) pair "{_name}" in reference '
                f'data file "{reference_file}" (from checkset spec '
                f'"{spec_file}").'
            )
        _reference_slice: pyam.IamDataFrame = reference.filter(
            variable=_variable, region=_region,
        )  # pyright: ignore[reportAssignmentType]
        _candidates: list[HistoricalComparisonOutput] = [
            HistoricalComparisonOutput(
                reference=_reference_slice,
                variable=_variable,
                region=_region,
                broadcast_dims=broadcast_dims,
            )
        ]
        for _alias_region in region_aliases.get(_region, []):
            _candidates.append(HistoricalComparisonOutput(
                reference=_reference_slice,
                variable=_variable,
                region=_region,
                broadcast_dims=broadcast_dims,
                fallback_from_region=_alias_region,
            ))
        if _region in region_fallbacks:
            _candidates.append(HistoricalComparisonOutput(
                reference=_reference_slice,
                variable=_variable,
                region=_region,
                broadcast_dims=broadcast_dims,
                include_pct_diff=False,
                fallback_from_region=region_fallbacks[_region],
            ))
        criteria[_name] = _candidates

    return MultiHistoricalComparisonOutput(criteria)
###END def _build_historical_comparison
