"""Build vetting check output objects from declarative YAML specs.

This module is the counterpart to `vetting_adapter.general_checks`: instead of
a fixed, hand-written check, it builds a check output object at runtime from a
small YAML spec plus a reference data file, both of which are expected to live
in a project's own definitions repository (see `vetting_adapter.profiles`).
This lets a project add its own checks (e.g., a harmonization check against
project-specific reference data) without writing any Python code or requiring
a release of this package.

Only one spec `type` is currently supported: `reference_comparison`, for
checks that compare a scenario's timeseries to a reference timeseries by
ratio (e.g., harmonization checks). See `build_check_from_spec` for the YAML
schema.
"""
from collections.abc import Callable
from pathlib import Path
import typing as tp

import numpy as np
import pyam
import yaml

from .criteria import ratio_reference_criterion
from .target_range import RatioTargetRange, RelativeRange
from .output.base import CriterionTargetRangeOutput, NoWriter
from .output.timeseries import (
    TimeseriesRefComparisonAndTargetOutput,
    TimeseriesRefFullComparisonOutput,
)



class CheckSpecError(ValueError):
    """Raised when a checkset YAML spec is missing or invalid."""
    ...
###END class CheckSpecError


SUPPORTED_CHECK_TYPES: tp.Final[tuple[str, ...]] = ('reference_comparison',)

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
) -> tuple[str, TimeseriesRefComparisonAndTargetOutput]:
    """Build a check output object from a checkset YAML spec file.

    Parameters
    ----------
    spec_file : pathlib.Path
        Path to the checkset YAML spec file.
    repo_root : pathlib.Path
        Root directory that paths given in the spec (such as the reference
        data file) are resolved relative to. This is normally the root of the
        cloned definitions repository that `spec_file` was found in.

    Returns
    -------
    (name, output) : tuple[str, TimeseriesRefComparisonAndTargetOutput]
        The check name (from the spec's `name` field, used as the key in the
        check registry), and the constructed output object.

    Raises
    ------
    CheckSpecError
        If the spec is missing required fields, refers to an unsupported
        `type`/`comparison.method`, or refers to a reference data file that
        does not exist.
    """
    spec: dict = load_spec(spec_file)
    check_type: str = spec.get('type', 'reference_comparison')
    if check_type not in SUPPORTED_CHECK_TYPES:
        raise CheckSpecError(
            f'Unsupported checkset type "{check_type}" in "{spec_file}". '
            f'Supported types: {", ".join(SUPPORTED_CHECK_TYPES)}.'
        )
    output: TimeseriesRefComparisonAndTargetOutput = \
        _build_reference_comparison(spec, repo_root, spec_file=spec_file)
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
