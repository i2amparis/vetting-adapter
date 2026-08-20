"""Built-in general-purpose vetting checks.

These are checks that are not specific to any one project and are therefore
made available to every validation profile by default (see
`vetting_adapter.profiles.get_available_checks`), regardless of whether the
profile's definitions repository declares any project-specific checks of its
own.

Module attributes
------------------
BUILTIN_CHECKS : dict[str, object]
    Mapping of built-in check name to its ready-to-use output object.
"""
from .ar6_vetting import ar6_vetting_target_range_output


BUILTIN_CHECKS: dict[str, object] = {
    'ar6_vetting': ar6_vetting_target_range_output,
}
