"""Project-agnostic machinery for defining and outputting vetting checks.

This subpackage contains no project-specific content -- no fixed check
definitions and no reference data. It provides the building blocks that both
the built-in checks in `vetting_adapter.general_checks` and declaratively
defined project-specific checks (see `vetting_adapter.core.spec`) are built
from.

Modules
-------
criteria
    `TimeseriesRefCriterion` and comparison-function helpers for comparing IAM
    output timeseries to a reference timeseries.
target_range
    `CriterionTargetRange` and subclasses for defining target values and
    acceptance ranges for criterion values.
spec
    Builds check output objects from declarative YAML specs, as used for
    project-specific checks bundled in a project's definitions repository.
dims
    Dimension name definitions used by `criteria`.

Subpackages
-----------
output
    Classes for producing and writing output (tables, styled tables, Excel
    files) from criteria and target ranges.
"""
