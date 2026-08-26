"""Tests for core.output.historical (diagnostic historical-comparison output)."""
import unittest

import pandas as pd
import pyam

from vetting_adapter.core.output.historical import (
    HistoricalComparisonOutput,
    MultiHistoricalComparisonOutput,
)


def _idf(rows: list[list], columns: list[str] = None) -> pyam.IamDataFrame:
    if columns is None:
        columns = ['model', 'scenario', 'region', 'variable', 'unit', 'year', 'value']
    return pyam.IamDataFrame(pd.DataFrame(rows, columns=columns))
###END def _idf


class TestHistoricalComparisonOutputExactRegion(unittest.TestCase):

    def setUp(self):
        self.reference = _idf([
            ['Historical', 'MS16', 'European Union (R9)', 'Population', 'million', 2019, 446.06],
            ['Historical', 'MS16', 'European Union (R9)', 'Population', 'million', 2020, 446.92],
        ])
        self.output = HistoricalComparisonOutput(
            reference=self.reference, variable='Population',
            region='European Union (R9)',
        )
        self.checked = _idf([
            ['modelA', 'scenA', 'European Union (R9)', 'Population', 'million', 2019, 440.0],
            ['modelA', 'scenA', 'European Union (R9)', 'Population', 'million', 2020, 450.0],
        ])
    ###END def setUp

    def test_is_applicable_true_when_overlap(self):
        self.assertTrue(self.output.is_applicable(self.checked))
    ###END def test_is_applicable_true_when_overlap

    def test_is_applicable_false_when_no_overlap(self):
        other = _idf([
            ['modelA', 'scenA', 'World', 'GDP|MER', 'billion USD_2017/yr', 2020, 1.0],
        ])
        self.assertFalse(self.output.is_applicable(other))
    ###END def test_is_applicable_false_when_no_overlap

    def test_prepare_output_values_and_pct_diff(self):
        result = self.output.prepare_output(self.checked)
        self.assertEqual(
            set(result.keys()), {'checked_values', 'historical_values', 'pct_diff'}
        )
        checked_vals = result['checked_values']
        hist_vals = result['historical_values']
        pct_diff = result['pct_diff']
        self.assertAlmostEqual(checked_vals[2019].iloc[0], 440.0)
        self.assertAlmostEqual(checked_vals[2020].iloc[0], 450.0)
        self.assertAlmostEqual(hist_vals[2019].iloc[0], 446.06)
        self.assertAlmostEqual(hist_vals[2020].iloc[0], 446.92)
        self.assertAlmostEqual(
            pct_diff[2019].iloc[0], (440.0 - 446.06) / 446.06 * 100.0
        )
        self.assertAlmostEqual(
            pct_diff[2020].iloc[0], (450.0 - 446.92) / 446.92 * 100.0
        )
    ###END def test_prepare_output_values_and_pct_diff

    def test_prepare_output_empty_when_not_applicable(self):
        other = _idf([
            ['modelA', 'scenA', 'World', 'GDP|MER', 'billion USD_2017/yr', 2020, 1.0],
        ])
        result = self.output.prepare_output(other)
        self.assertTrue(result['checked_values'].empty)
        self.assertTrue(result['historical_values'].empty)
        self.assertTrue(result['pct_diff'].empty)
    ###END def test_prepare_output_empty_when_not_applicable

    def test_prepare_output_ignores_model_name(self):
        """Checks are model-agnostic: any model/scenario should compare."""
        different_model = _idf([
            ['some_other_model', 'some_other_scenario', 'European Union (R9)',
             'Population', 'million', 2019, 500.0],
        ])
        result = self.output.prepare_output(different_model)
        self.assertAlmostEqual(result['checked_values'][2019].iloc[0], 500.0)
    ###END def test_prepare_output_ignores_model_name

###END class TestHistoricalComparisonOutputExactRegion


class TestHistoricalComparisonOutputFallbackRegion(unittest.TestCase):

    def setUp(self):
        self.reference = _idf([
            ['Historical', 'MS16', 'EU27+UK+NO+CH', 'Production|Chemicals|Plastics',
             'Mt/yr', 2019, 57.9],
        ])
        self.exact = HistoricalComparisonOutput(
            reference=self.reference, variable='Production|Chemicals|Plastics',
            region='EU27+UK+NO+CH',
        )
        self.fallback = HistoricalComparisonOutput(
            reference=self.reference, variable='Production|Chemicals|Plastics',
            region='EU27+UK+NO+CH',
            fallback_from_region='European Union (R9)',
            include_pct_diff=False,
        )
        self.checked_eu27_only = _idf([
            ['modelA', 'scenA', 'European Union (R9)', 'Production|Chemicals|Plastics',
             'Mt/yr', 2019, 55.0],
        ])
    ###END def setUp

    def test_exact_not_applicable_for_eu27_only_data(self):
        self.assertFalse(self.exact.is_applicable(self.checked_eu27_only))
    ###END def test_exact_not_applicable_for_eu27_only_data

    def test_fallback_applicable_for_eu27_only_data(self):
        self.assertTrue(self.fallback.is_applicable(self.checked_eu27_only))
    ###END def test_fallback_applicable_for_eu27_only_data

    def test_fallback_output_has_no_pct_diff(self):
        result = self.fallback.prepare_output(self.checked_eu27_only)
        self.assertEqual(set(result.keys()), {'checked_values', 'historical_values'})
        self.assertAlmostEqual(result['checked_values'][2019].iloc[0], 55.0)
        self.assertAlmostEqual(result['historical_values'][2019].iloc[0], 57.9)
    ###END def test_fallback_output_has_no_pct_diff

    def test_fallback_not_applicable_when_region_entirely_absent(self):
        other = _idf([
            ['modelA', 'scenA', 'World', 'GDP|MER', 'billion USD_2017/yr', 2020, 1.0],
        ])
        self.assertFalse(self.fallback.is_applicable(other))
    ###END def test_fallback_not_applicable_when_region_entirely_absent

###END class TestHistoricalComparisonOutputFallbackRegion


class TestMultiHistoricalComparisonOutput(unittest.TestCase):

    def setUp(self):
        plastics_reference = _idf([
            ['Historical', 'MS16', 'EU27+UK+NO+CH', 'Production|Chemicals|Plastics',
             'Mt/yr', 2019, 57.9],
        ])
        exact = HistoricalComparisonOutput(
            reference=plastics_reference, variable='Production|Chemicals|Plastics',
            region='EU27+UK+NO+CH',
        )
        fallback = HistoricalComparisonOutput(
            reference=plastics_reference, variable='Production|Chemicals|Plastics',
            region='EU27+UK+NO+CH',
            fallback_from_region='European Union (R9)',
            include_pct_diff=False,
        )

        pop_reference = _idf([
            ['Historical', 'MS16', 'European Union (R9)', 'Population', 'million',
             2019, 446.06],
        ])
        pop_exact = HistoricalComparisonOutput(
            reference=pop_reference, variable='Population',
            region='European Union (R9)',
        )
        pop_alias = HistoricalComparisonOutput(
            reference=pop_reference, variable='Population',
            region='European Union (R9)',
            fallback_from_region='EU27',
            include_pct_diff=True,
        )

        self.multi = MultiHistoricalComparisonOutput({
            'plastics': [exact, fallback],
            'population': [pop_exact, pop_alias],
        })
    ###END def setUp

    def test_returns_exact_when_available(self):
        checked = _idf([
            ['modelA', 'scenA', 'EU27+UK+NO+CH', 'Production|Chemicals|Plastics',
             'Mt/yr', 2019, 55.0],
        ])
        output = self.multi.get_applicable_output('plastics', checked)
        self.assertIsNotNone(output)
        self.assertTrue(output.include_pct_diff)
        self.assertIsNone(output.fallback_from_region)
    ###END def test_returns_exact_when_available

    def test_falls_back_when_exact_region_missing(self):
        checked = _idf([
            ['modelA', 'scenA', 'European Union (R9)', 'Production|Chemicals|Plastics',
             'Mt/yr', 2019, 55.0],
        ])
        output = self.multi.get_applicable_output('plastics', checked)
        self.assertIsNotNone(output)
        self.assertFalse(output.include_pct_diff)
    ###END def test_falls_back_when_exact_region_missing

    def test_returns_none_when_neither_applicable(self):
        checked = _idf([
            ['modelA', 'scenA', 'World', 'GDP|MER', 'billion USD_2017/yr', 2020, 1.0],
        ])
        output = self.multi.get_applicable_output('plastics', checked)
        self.assertIsNone(output)
    ###END def test_returns_none_when_neither_applicable

    def test_region_alias_still_computes_pct_diff(self):
        """An alias match (same region, different name) is not the same as
        a fallback match (different, broader/narrower region): percent
        difference should still be included."""
        checked = _idf([
            ['modelA', 'scenA', 'EU27', 'Population', 'million', 2019, 440.0],
        ])
        output = self.multi.get_applicable_output('population', checked)
        self.assertIsNotNone(output)
        self.assertTrue(output.include_pct_diff)
        tables = output.prepare_output(checked)
        self.assertIn('pct_diff', tables)
        self.assertFalse(tables['pct_diff'].empty)
    ###END def test_region_alias_still_computes_pct_diff

    def test_exact_region_preferred_over_alias(self):
        checked = _idf([
            ['modelA', 'scenA', 'European Union (R9)', 'Population', 'million',
             2019, 440.0],
        ])
        output = self.multi.get_applicable_output('population', checked)
        self.assertIsNotNone(output)
        self.assertIsNone(output.fallback_from_region)
    ###END def test_exact_region_preferred_over_alias

###END class TestMultiHistoricalComparisonOutput
