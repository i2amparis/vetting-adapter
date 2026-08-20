# vetting-adapter

`vetting-adapter` is a Python package for running vetting/feasibility checks
(e.g. IPCC AR6 vetting, harmonization checks against reference data) on
Integrated Assessment Model (IAM) results in IAMC format. It builds on
[`pathways-ensemble-analysis`](https://gitlab.com/climateanalytics/pathways-ensemble-analysis)
and adds a project-profile layer, so one validation UI can switch between
different sets of vetting checks depending on the project whose results are
being assessed.

The package is primarily used by
[`i2amparis/validation-ui`](https://github.com/i2amparis/validation-ui), but
it can also be imported directly by scripts or notebooks that only need the
vetting checks (e.g. for exploratory analysis), without any dependency on the
nomenclature/name-validation side of the stack.

This package started as [CICERO&#39;s `iamcompact-vetting`](https://github.com/ciceroOslo/iamcompact-vetting),
built specifically for the HORIZON EUROPE project IAM COMPACT (see `NOTICE`
for funding attribution), and has since been generalized so that project-wide
checks like IPCC AR6 vetting ship as built-ins. Project-specific checks
(like the original IAM COMPACT GDP/population harmonization check) are
declarative specs owned by each project's own definitions repository, rather
than being hardcoded into this package.

## What This Repository Contains

- `vetting_adapter/core/`: project-agnostic machinery -- criteria for
  comparing IAM output timeseries to a reference (`core/criteria.py`),
  target/range definitions (`core/target_range.py`), output/Excel-writing
  classes (`core/output/`), and a builder for declarative,
  YAML-defined checks (`core/spec.py`). Contains no fixed check definitions
  or reference data of its own.
- `vetting_adapter/general_checks/`: built-in checks that are not specific to
  any one project and are therefore available to every validation profile by
  default. Currently just the IPCC AR6 vetting criteria
  (`general_checks/ar6_vetting.py`).
- `vetting_adapter/profiles.py`: resolves which vetting checks are available
  for a given validation profile, by reading the same profile manifest that
  [`nomenclature-adapter`](https://github.com/i2amparis/nomenclature-adapter)
  uses (see `get_available_checks`).

Project-specific checks (e.g. a GDP/population harmonization check against
one project's own reference data) are *not* bundled in this package. Instead,
they are declarative YAML specs (parsed by `core/spec.py`) that live in the
project's own nomenclature definitions repository, alongside its
`definitions/` and `mappings/` folders, and are referenced from that
project's profile manifest under a `vetting:` key. See
`vetting_adapter.core.spec.build_check_from_spec` for the spec schema, and
e.g. `TRANSIENCE-MIC3/transience-nomenclature-definitions`'s `vetting/`
folder for a worked example.

## Installation

Install directly from GitHub with `pip`:

```bash
pip install git+https://github.com/i2amparis/vetting-adapter.git
```

With `uv`, add the package to another project as a Git dependency:

```bash
uv add "vetting-adapter @ git+https://github.com/i2amparis/vetting-adapter.git"
```

## Usage

Get the vetting checks available for a validation profile (built-in checks
like AR6 vetting are always included; project-specific checks are added if
the profile declares any):

```python
import vetting_adapter as ivet

checks = ivet.get_available_checks(profile_name="transience")
print(checks.keys())  # e.g. dict_keys(['ar6_vetting', 'gdp_pop_harmonization'])

ar6_output = checks["ar6_vetting"]
styled = ar6_output.prepare_styled_output(iamdf, prepare_output_kwargs=dict(add_summary_output=True))
```

## Development

Install the package in editable mode from this repository:

```bash
uv sync
```

Run tests, if present, with:

```bash
uv run pytest
```

## License

This package is released under [Apache license](./LICENSE). See
[3rd-party-licenses](./3rd-party-licenses/) for licensing information about
source code from other packages used in this package.
