"""Resolve which vetting checks are available for a validation profile.

This module reads the *same* profile manifest that `nomenclature_adapter`
uses to resolve definitions and region mappings (see the `profiles/*.yaml`
files in that package), interpreting one additional, optional top-level key
that `nomenclature_adapter` itself ignores:

    vetting:
      checks:
        - builtin: ar6_vetting
        - repository: transience-defs
          file: vetting/gdp_pop_harmonization.yaml

Each `repository`/`file` entry points at a checkset YAML spec (see
`vetting_adapter.core.spec`) inside one of the profile's own definitions
repositories, e.g. under a `vetting/` folder alongside `definitions/` and
`mappings/`.

The built-in general checks (see `vetting_adapter.general_checks`) are always
included. A profile with no `vetting:` section, a repository/file that is not
found, or a checkset spec that fails to parse, simply falls back to (or stays
at) the built-in checks -- none of these are treated as errors, since a
project is not required to define any checks of its own.
"""
import logging
from pathlib import Path
import typing as tp

import nomenclature_adapter as nomadapter

from . import general_checks
from .core.spec import build_check_from_spec, CheckSpecError


logger: logging.Logger = logging.getLogger(__name__)


def get_available_checks(
        profile_name: tp.Optional[str] = None,
) -> dict[str, tp.Any]:
    """Return the vetting checks available for a validation profile.

    Parameters
    ----------
    profile_name : str, optional
        Name of the validation profile (as used by `nomenclature_adapter`).
        Optional, defaults to `nomenclature_adapter.DEFAULT_PROFILE`.

    Returns
    -------
    dict[str, object]
        Mapping of check name to its output object (an instance of
        `MultiCriterionTargetRangeOutput`,
        `TimeseriesRefComparisonAndTargetOutput`, or similar, depending on the
        check). Always includes the built-in checks from
        `vetting_adapter.general_checks.BUILTIN_CHECKS`; may include
        additional project-specific checks if the profile declares any (see
        module docstring).
    """
    if profile_name is None:
        profile_name = nomadapter.DEFAULT_PROFILE

    checks: dict[str, tp.Any] = dict(general_checks.BUILTIN_CHECKS)

    try:
        manifest: dict = nomadapter.get_profile_manifest(profile_name)
    except FileNotFoundError:
        logger.warning(
            'No profile manifest found for "%s"; using built-in vetting '
            'checks only.',
            profile_name,
        )
        return checks

    vetting_cfg: dict|None = manifest.get('vetting')
    if not vetting_cfg:
        return checks

    for entry in vetting_cfg.get('checks', []):
        if not isinstance(entry, dict):
            logger.warning(
                'Ignoring invalid vetting check entry in profile "%s": %r',
                profile_name, entry,
            )
            continue
        if 'builtin' in entry:
            _add_builtin_check(checks, entry['builtin'], profile_name)
            continue
        _add_declarative_check(checks, entry, profile_name)

    return checks
###END def get_available_checks


def _add_builtin_check(
        checks: dict[str, tp.Any],
        builtin_name: str,
        profile_name: str,
) -> None:
    if builtin_name not in general_checks.BUILTIN_CHECKS:
        logger.warning(
            'Unknown built-in check "%s" referenced in profile "%s"; '
            'ignoring.',
            builtin_name, profile_name,
        )
        return
    checks[builtin_name] = general_checks.BUILTIN_CHECKS[builtin_name]
###END def _add_builtin_check


def _add_declarative_check(
        checks: dict[str, tp.Any],
        entry: dict,
        profile_name: str,
) -> None:
    repo_name: str|None = entry.get('repository')
    spec_rel_path: str|None = entry.get('file')
    if not repo_name or not spec_rel_path:
        logger.warning(
            'Ignoring invalid vetting check entry in profile "%s": %r '
            '(expected "repository" and "file", or "builtin").',
            profile_name, entry,
        )
        return

    try:
        repo_root: Path = nomadapter.get_profile_repo_path(
            repo_name=repo_name, profile_name=profile_name
        )
    except FileNotFoundError:
        logger.warning(
            'Repository "%s" not available for profile "%s"; skipping '
            'vetting check "%s".',
            repo_name, profile_name, spec_rel_path,
        )
        return

    spec_file: Path = repo_root / spec_rel_path
    if not spec_file.is_file():
        logger.warning(
            'Vetting checkset file "%s" not found in repository "%s" for '
            'profile "%s"; skipping.',
            spec_rel_path, repo_name, profile_name,
        )
        return

    try:
        check_name, output = build_check_from_spec(spec_file, repo_root)
    except CheckSpecError as _err:
        logger.warning(
            'Invalid vetting checkset "%s" for profile "%s": %s',
            spec_file, profile_name, _err,
        )
        return

    checks[check_name] = output
###END def _add_declarative_check
