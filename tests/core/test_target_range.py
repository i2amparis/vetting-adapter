"""Tests for CriterionTargetRange's handling of inapplicable criteria.

A production incident showed that if even one criterion in a
`MultiCriterionTargetRangeOutput` (e.g. the built-in AR6 vetting checkset)
had no matching variable/region/year data at all in the uploaded dataset,
`pathways_ensemble_analysis` would raise, aborting computation of every other
criterion -- even ones that *did* have matching data. These tests lock in the
fix: `CriterionTargetRange.get_values` now returns an empty Series instead of
raising in that situation, which lets `MultiCriterionTargetRangeOutput`
compute all applicable criteria and simply mark the inapplicable ones as "not
assessed" (the same status already used for individual missing model/scenario
combinations), instead of the whole output aborting.
"""
import unittest
import unittest.mock

import pandas as pd
import pyam
from pathways_ensemble_analysis.criteria.base import (
    ChangeOverTimeCriterion,
    SingleVariableCriterion,
)

from vetting_adapter.core.target_range import CriterionTargetRange
from vetting_adapter.core.output.base import (
    CTCol,
    MultiCriterionTargetRangeOutput,
    NoWriter,
)


def _make_iamdf() -> pyam.IamDataFrame:
    return pyam.IamDataFrame(pd.DataFrame([
        {
            'model': 'M1', 'scenario': 'S1', 'region': 'World',
            'variable': 'Emissions|CO2', 'unit': 'Mt CO2/yr', 2020: 44000.0,
        },
    ]))
###END def _make_iamdf


def _make_criterion(
        *,
        name: str = 'CO2 2020',
        variable: str = 'Emissions|CO2',
        region: str = 'World',
        year: int = 2020,
) -> CriterionTargetRange:
    return CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name=name,
            region=region,
            year=year,
            variable=variable,
            unit='Mt CO2 / yr',
        ),
        target=44251.0,
        unit='Mt CO2 / yr',
        range=(0.0, 100000.0),
    )
###END def _make_criterion


class TestCriterionTargetRangeApplicability(unittest.TestCase):

    def test_get_values_returns_empty_series_for_missing_variable(self):
        criterion = _make_criterion(variable='Emissions|CH4')
        values = criterion.get_values(_make_iamdf())
        self.assertTrue(values.empty)
    ###END def test_get_values_returns_empty_series_for_missing_variable

    def test_get_values_returns_empty_series_for_mismatched_region(self):
        # Reproduces the actual production case: a dataset using "WORLD"
        # (or any region name other than exactly "World") for the AR6
        # checks' hardcoded region="World" filter.
        criterion = _make_criterion(region='WORLD')
        values = criterion.get_values(_make_iamdf())
        self.assertTrue(values.empty)
    ###END def test_get_values_returns_empty_series_for_mismatched_region

    def test_get_values_still_works_normally_when_applicable(self):
        criterion = _make_criterion()
        values = criterion.get_values(_make_iamdf())
        self.assertFalse(values.empty)
        self.assertEqual(values.iloc[0], 44000.0)
    ###END def test_get_values_still_works_normally_when_applicable

    def test_get_values_still_raises_for_unrelated_value_errors(self):
        # A ValueError not matching the "not available" pattern should still
        # propagate -- this fix must not silently swallow real bugs.
        criterion = _make_criterion()
        with unittest.mock.patch.object(
                criterion._criterion, 'get_values',
                side_effect=ValueError('something else went wrong'),
        ):
            with self.assertRaises(ValueError):
                criterion.get_values(_make_iamdf())
    ###END def test_get_values_still_raises_for_unrelated_value_errors

    def test_is_applicable(self):
        applicable = _make_criterion()
        inapplicable = _make_criterion(variable='Emissions|CH4')
        iamdf = _make_iamdf()
        self.assertTrue(applicable.is_applicable(iamdf))
        self.assertFalse(inapplicable.is_applicable(iamdf))
    ###END def test_is_applicable

    def test_describe_requirements(self):
        criterion = _make_criterion()
        description = criterion.describe_requirements()
        self.assertEqual(description['name'], 'CO2 2020')
        self.assertEqual(description['variable'], 'Emissions|CO2')
        self.assertEqual(description['region'], 'World')
        self.assertEqual(description['year'], 2020)
        self.assertEqual(description['unit'], 'Mt CO2 / yr')
        self.assertEqual(description['target'], 44251.0)
        self.assertEqual(description['range'], (0.0, 100000.0))
    ###END def test_describe_requirements

###END class TestCriterionTargetRangeApplicability


class TestChangeOverTimeCriterionMissingReferenceYear(unittest.TestCase):
    """Regression test for a second, distinct failure mode found while
    testing against a real uploaded dataset: `ChangeOverTimeCriterion` (used
    by the AR6 "... 2010-2020 change" criterion) selects *two* years at once
    (`year` and `reference_year`). `select_vars` only raises the
    `ValueError` handled by `test_get_values_returns_empty_series_for_*`
    above if the combined filter is entirely empty -- if the variable is
    present for `year` but `reference_year` doesn't exist in the data at
    all (e.g. a dataset that only starts in 2017, with no 2010 column), the
    filter succeeds (since 2020 matches) and a later
    `sel.timeseries()[reference_year]` lookup instead raises a bare
    `KeyError`, uncaught by the ValueError handling alone.
    """

    def _make_change_criterion(self) -> CriterionTargetRange:
        return CriterionTargetRange(
            criterion=ChangeOverTimeCriterion(
                criterion_name='CO2 change',
                region='World',
                year=2020,
                reference_year=2010,
                variable='Emissions|CO2',
            ),
            target=0.25,
            range=(0.0, 0.5),
        )
    ###END def _make_change_criterion

    def test_missing_reference_year_returns_empty_series(self):
        # `year` (2020) is present, but `reference_year` (2010) is not in
        # the data at all.
        iamdf = pyam.IamDataFrame(pd.DataFrame([
            {
                'model': 'M1', 'scenario': 'S1', 'region': 'World',
                'variable': 'Emissions|CO2', 'unit': 'Mt CO2/yr', 2020: 44000.0,
            },
        ]))
        criterion = self._make_change_criterion()
        values = criterion.get_values(iamdf)
        self.assertTrue(values.empty)
        self.assertFalse(criterion.is_applicable(iamdf))
    ###END def test_missing_reference_year_returns_empty_series

    def test_works_normally_when_both_years_present(self):
        iamdf = pyam.IamDataFrame(pd.DataFrame([
            {
                'model': 'M1', 'scenario': 'S1', 'region': 'World',
                'variable': 'Emissions|CO2', 'unit': 'Mt CO2/yr',
                2010: 40000.0, 2020: 44000.0,
            },
        ]))
        criterion = self._make_change_criterion()
        values = criterion.get_values(iamdf)
        self.assertFalse(values.empty)
        self.assertAlmostEqual(values.iloc[0], 0.1)
    ###END def test_works_normally_when_both_years_present

###END class TestChangeOverTimeCriterionMissingReferenceYear


class TestMultiCriterionTargetRangeOutputPartialApplicability(unittest.TestCase):
    """One inapplicable criterion must not prevent computing the others."""

    def setUp(self):
        self.applicable = _make_criterion(name='applicable')
        self.inapplicable = _make_criterion(
            name='inapplicable', variable='Emissions|CH4',
        )
        self.outputter = MultiCriterionTargetRangeOutput(
            criteria={
                self.applicable.name: self.applicable,
                self.inapplicable.name: self.inapplicable,
            },
            writer=NoWriter(),
            columns=[CTCol.INRANGE, CTCol.VALUE],
            column_titles={CTCol.INRANGE: 'Passed', CTCol.VALUE: 'Value'},
            summary_keys={
                CTCol.INRANGE: 'Pass vs. Fail Summary',
                CTCol.VALUE: 'Values Summary',
            },
        )
    ###END def setUp

    def test_prepare_output_does_not_raise(self):
        output = self.outputter.prepare_output(
            _make_iamdf(), add_summary_output=True,
        )
        self.assertIn('applicable', output)
        self.assertIn('inapplicable', output)
    ###END def test_prepare_output_does_not_raise

    def test_summary_shows_applicable_value_and_nan_for_inapplicable(self):
        output = self.outputter.prepare_output(
            _make_iamdf(), add_summary_output=True,
        )
        values_summary: pd.DataFrame = output['Values Summary']
        self.assertEqual(
            values_summary.loc[('M1', 'S1'), 'applicable'], 44000.0,
        )
        self.assertTrue(
            pd.isna(values_summary.loc[('M1', 'S1'), 'inapplicable'])
        )
    ###END def test_summary_shows_applicable_value_and_nan_for_inapplicable

    def test_prepare_styled_output_does_not_raise(self):
        # Regression test for the full pipeline used by the AR6 vetting
        # page (`prepare_styled_output` -> `style_output`), which indexes
        # into the prepared-output dict by criteria name and would raise a
        # KeyError if inapplicable criteria were dropped from that dict
        # instead of being included with empty/NaN results.
        styled = self.outputter.prepare_styled_output(
            _make_iamdf(),
            prepare_output_kwargs=dict(add_summary_output=True),
            style_output_kwargs=dict(include_summary=True),
        )
        self.assertIn('applicable', styled)
        self.assertIn('inapplicable', styled)
    ###END def test_prepare_styled_output_does_not_raise

    def test_inapplicable_criterion_styled_output_exports_to_excel(self):
        # Regression test: `Styler.map_index(..., axis=0)` on a zero-row
        # DataFrame (exactly what an inapplicable criterion's per-criterion
        # sheet now is) makes pandas' own `Styler.to_excel` raise a spurious
        # KeyError, reproduced directly against a trivial empty-index Styler
        # unrelated to this package's logic. `apply_common_styling` skips
        # that call when there are no rows to style; this locks the fix in
        # against the real xlsxwriter export path used for the Excel
        # download button.
        import io
        styled = self.outputter.prepare_styled_output(
            _make_iamdf(),
            prepare_output_kwargs=dict(add_summary_output=True),
            style_output_kwargs=dict(include_summary=True),
        )
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            styled['inapplicable'].to_excel(
                writer, sheet_name='inapplicable', merge_cells=False,
            )
        self.assertGreater(buf.tell(), 0)
    ###END def test_inapplicable_criterion_styled_output_exports_to_excel

###END class TestMultiCriterionTargetRangeOutputPartialApplicability


class TestMultiCriterionTargetRangeOutputAllInapplicable(unittest.TestCase):
    """No criterion applicable: must not raise, but the caller needs to be
    able to tell the result apart from "everything passed" (see the
    `len(df) == 0` guard added in validation-ui's AR6 vetting page)."""

    def test_prepare_output_returns_empty_summary_without_raising(self):
        criterion = _make_criterion(region='WORLD')
        outputter = MultiCriterionTargetRangeOutput(
            criteria={criterion.name: criterion},
            writer=NoWriter(),
            columns=[CTCol.INRANGE, CTCol.VALUE],
            column_titles={CTCol.INRANGE: 'Passed', CTCol.VALUE: 'Value'},
            summary_keys={
                CTCol.INRANGE: 'Pass vs. Fail Summary',
                CTCol.VALUE: 'Values Summary',
            },
        )
        output = outputter.prepare_output(
            _make_iamdf(), add_summary_output=True,
        )
        self.assertEqual(len(output['Values Summary']), 0)
    ###END def test_prepare_output_returns_empty_summary_without_raising

###END class TestMultiCriterionTargetRangeOutputAllInapplicable


if __name__ == '__main__':
    unittest.main()
