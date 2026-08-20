"""Project-agnostic vetting/assessment checks for IAMC-format model results.

Provides a project-agnostic engine (`core`), built-in general checks that
apply to any project (`general_checks`), and a profile-driven registry
(`get_available_checks`) that adds project-specific checks declared in a
project's own definitions repository. See `vetting_adapter.profiles` for
details of how project-specific checks are resolved.
"""
from . import core
from . import general_checks
from .profiles import get_available_checks


__all__ = [
    'core',
    'general_checks',
    'get_available_checks',
]
