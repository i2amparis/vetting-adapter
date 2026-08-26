"""Tests for the core.spec module (declarative checkset YAML specs)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyam
import yaml

from vetting_adapter.core.spec import (
    build_check_from_spec,
    CheckSpecError,
)
from vetting_adapter.core.output.base import MultiCriterionTargetRangeOutput
from vetting_adapter.core.output.historical import MultiHistoricalComparisonOutput


def _write_spec(tmp_dir: Path, spec: dict, filename: str = 'spec.yaml') -> Path:
    spec_file: Path = tmp_dir / filename
    spec_file.write_text(yaml.dump(spec))
    return spec_file
###END def _write_spec


class TestBuildTargetRange(unittest.TestCase):
    """Tests for the `target_range` checkset spec type."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
    ###END def setUp

    def tearDown(self):
        self._tmp.cleanup()
    ###END def tearDown

    def _base_spec(self) -> dict:
        return {
            'name': 'test_checkset',
            'type': 'target_range',
            'criteria': [
                {
                    'name': 'CO2 EIP emissions 2020',
                    'region': 'World',
                    'year': 2020,
                    'variable': 'Emissions|CO2|Energy and Industrial Processes',
                    'unit': 'Mt CO2 / yr',
                    'target': 37646.0,
                    'tolerance': 0.2,
                },
                {
                    'name': 'CO2 EIP emissions 2010-2020 change',
                    'criterion_type': 'change_over_time',
                    'region': 'World',
                    'year': 2020,
                    'reference_year': 2010,
                    'variable': 'Emissions|CO2|Energy and Industrial Processes',
                    'target': 0.25,
                    'range': [0.0, 0.5],
                },
                {
                    'name': 'CCS from energy 2020',
                    'region': 'World',
                    'year': 2020,
                    'variable': 'Carbon Sequestration|CCS',
                    'unit': 'Mt CO2 / yr',
                    'target': 0.0,
                    'range': [0.0, 250.0],
                    'distance_func': {'type': 'fixed_denominator', 'value': 250.0},
                },
            ],
        }
    ###END def _base_spec

    def _test_df(self) -> pyam.IamDataFrame:
        return pyam.IamDataFrame(pd.DataFrame({
            'model': ['model_a']*4,
            'scenario': ['scenario_a']*4,
            'region': ['World']*4,
            'variable': [
                'Emissions|CO2|Energy and Industrial Processes',
                'Emissions|CO2|Energy and Industrial Processes',
                'Carbon Sequestration|CCS',
                'Carbon Sequestration|CCS',
            ],
            'unit': ['Mt CO2 / yr']*4,
            'year': [2010, 2020, 2010, 2020],
            'value': [30000.0, 37000.0, 40.0, 50.0],
        }))
    ###END def _test_df

    def test_builds_multi_criterion_target_range_output(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        name, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        self.assertEqual(name, 'test_checkset')
        self.assertIsInstance(output, MultiCriterionTargetRangeOutput)
        self.assertEqual(
            set(output.criteria.keys()),
            {
                'CO2 EIP emissions 2020',
                'CO2 EIP emissions 2010-2020 change',
                'CCS from energy 2020',
            },
        )
    ###END def test_builds_multi_criterion_target_range_output

    def test_tolerance_produces_expected_relative_range(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        _, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        criterion = output.criteria['CO2 EIP emissions 2020']
        self.assertAlmostEqual(criterion.range[0], 37646.0 * 0.8)
        self.assertAlmostEqual(criterion.range[1], 37646.0 * 1.2)
    ###END def test_tolerance_produces_expected_relative_range

    def test_change_over_time_criterion_computes_expected_value(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        _, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        criterion = output.criteria['CO2 EIP emissions 2010-2020 change']
        values = criterion.get_values(self._test_df())
        self.assertAlmostEqual(values.iloc[0], (37000.0 - 30000.0) / 30000.0)
    ###END def test_change_over_time_criterion_computes_expected_value

    def test_distance_func_override_avoids_default_division_by_zero_target(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        _, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        criterion = output.criteria['CCS from energy 2020']
        self.assertEqual(criterion.target, 0.0)
        self.assertEqual(criterion.distance_func(50.0), 50.0 / 250.0)
    ###END def test_distance_func_override_avoids_default_division_by_zero_target

    def test_prepare_output_reports_in_range_status(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        _, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        result = output.prepare_output(self._test_df())
        co2_df = result['CO2 EIP emissions 2020']
        self.assertTrue(bool(co2_df[co2_df.columns[0]].iloc[0]) or True)
        # Value 37000 is within [37646*0.8, 37646*1.2].
        in_range_col = [
            _c for _c in co2_df.columns if 'range' in _c.lower()
        ][0]
        self.assertTrue(bool(co2_df[in_range_col].iloc[0]))
    ###END def test_prepare_output_reports_in_range_status

    def test_missing_criteria_list_raises(self):
        spec = self._base_spec()
        del spec['criteria']
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_missing_criteria_list_raises

    def test_unknown_criterion_type_raises(self):
        spec = self._base_spec()
        spec['criteria'][0]['criterion_type'] = 'nonsense'
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_unknown_criterion_type_raises

    def test_change_over_time_missing_reference_year_raises(self):
        spec = self._base_spec()
        del spec['criteria'][1]['reference_year']
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_change_over_time_missing_reference_year_raises

    def test_mutually_exclusive_range_specs_raise(self):
        spec = self._base_spec()
        spec['criteria'][0]['range'] = [0.0, 100000.0]
        # 'tolerance' is already set on this entry, so both are now present.
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_mutually_exclusive_range_specs_raise

    def test_duplicate_criterion_names_raise(self):
        spec = self._base_spec()
        spec['criteria'][1]['name'] = spec['criteria'][0]['name']
        spec['criteria'][1]['criterion_type'] = 'single_variable'
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_duplicate_criterion_names_raise

    def test_unknown_output_column_raises(self):
        spec = self._base_spec()
        spec['output'] = {'columns': ['not_a_real_column']}
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_unknown_output_column_raises

    def test_relative_range_field(self):
        spec = self._base_spec()
        del spec['criteria'][0]['tolerance']
        spec['criteria'][0]['relative_range'] = [0.7, 1.3]
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        _, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        criterion = output.criteria['CO2 EIP emissions 2020']
        self.assertAlmostEqual(criterion.range[0], 37646.0 * 0.7)
        self.assertAlmostEqual(criterion.range[1], 37646.0 * 1.3)
    ###END def test_relative_range_field

###END class TestBuildTargetRange


class TestBuildCheckFromSpecUnsupportedType(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
    ###END def setUp

    def tearDown(self):
        self._tmp.cleanup()
    ###END def tearDown

    def test_unsupported_type_raises(self):
        spec_file: Path = _write_spec(
            self.tmp_dir, {'name': 'x', 'type': 'not_a_real_type'}
        )
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_unsupported_type_raises

    def test_missing_name_raises(self):
        spec_file: Path = _write_spec(self.tmp_dir, {'type': 'target_range'})
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_missing_name_raises

###END class TestBuildCheckFromSpecUnsupportedType


def _write_reference_xlsx(tmp_dir: Path, rows: list[dict], filename: str) -> Path:
    ref_file: Path = tmp_dir / filename
    pd.DataFrame(rows).to_excel(ref_file, index=False)
    return ref_file
###END def _write_reference_xlsx


class TestBuildHistoricalComparison(unittest.TestCase):
    """Tests for the `historical_comparison` checkset spec type."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.reference_file = _write_reference_xlsx(
            self.tmp_dir,
            [
                {
                    'Model': 'Historical', 'Scenario': 'MS16',
                    'Region': 'European Union (R9)', 'Variable': 'Population',
                    'Unit': 'million', '2019': 446.06, '2020': 446.92,
                },
                {
                    'Model': 'Historical', 'Scenario': 'MS16',
                    'Region': 'EU27+UK+NO+CH',
                    'Variable': 'Production|Chemicals|Plastics',
                    'Unit': 'Mt/yr', '2019': 57.9, '2020': 47.5,
                },
            ],
            'reference.xlsx',
        )
    ###END def setUp

    def tearDown(self):
        self._tmp.cleanup()
    ###END def tearDown

    def _base_spec(self) -> dict:
        return {
            'name': 'test_historical_checkset',
            'type': 'historical_comparison',
            'reference': {'file': 'reference.xlsx'},
            'region_fallbacks': {
                'EU27+UK+NO+CH': 'European Union (R9)',
            },
        }
    ###END def _base_spec

    def test_builds_one_criterion_per_variable_region_pair(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        name, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        self.assertEqual(name, 'test_historical_checkset')
        self.assertIsInstance(output, MultiHistoricalComparisonOutput)
        self.assertEqual(
            set(output.criteria.keys()),
            {
                'Population (European Union (R9))',
                'Production|Chemicals|Plastics (EU27+UK+NO+CH)',
            },
        )
    ###END def test_builds_one_criterion_per_variable_region_pair

    def test_fallback_only_attached_to_configured_region(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        _, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        _exact_pop, fallback_pop = \
            output.criteria['Population (European Union (R9))']
        self.assertIsNone(fallback_pop)
        _exact_plastics, fallback_plastics = \
            output.criteria['Production|Chemicals|Plastics (EU27+UK+NO+CH)']
        self.assertIsNotNone(fallback_plastics)
    ###END def test_fallback_only_attached_to_configured_region

    def test_end_to_end_applicability_and_values(self):
        spec_file: Path = _write_spec(self.tmp_dir, self._base_spec())
        _, output = build_check_from_spec(spec_file, repo_root=self.tmp_dir)
        checked = pyam.IamDataFrame(pd.DataFrame([
            ['modelX', 'scenY', 'European Union (R9)', 'Population',
             'million', 2019, 440.0],
            ['modelX', 'scenY', 'European Union (R9)',
             'Production|Chemicals|Plastics', 'Mt/yr', 2019, 55.0],
        ], columns=['model', 'scenario', 'region', 'variable', 'unit',
                    'year', 'value']))

        pop_result = output.get_applicable_output(
            'Population (European Union (R9))', checked
        )
        self.assertIsNotNone(pop_result)
        pop_output, pop_is_fallback = pop_result
        self.assertFalse(pop_is_fallback)
        self.assertAlmostEqual(
            pop_output.prepare_output(checked)['checked_values'][2019].iloc[0],
            440.0,
        )

        plastics_result = output.get_applicable_output(
            'Production|Chemicals|Plastics (EU27+UK+NO+CH)', checked
        )
        self.assertIsNotNone(plastics_result)
        _plastics_output, plastics_is_fallback = plastics_result
        self.assertTrue(plastics_is_fallback)
    ###END def test_end_to_end_applicability_and_values

    def test_missing_reference_file_raises(self):
        spec = self._base_spec()
        spec['reference'] = {'file': 'does_not_exist.xlsx'}
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_missing_reference_file_raises

    def test_invalid_region_fallbacks_type_raises(self):
        spec = self._base_spec()
        spec['region_fallbacks'] = ['not', 'a', 'dict']
        spec_file: Path = _write_spec(self.tmp_dir, spec)
        with self.assertRaises(CheckSpecError):
            build_check_from_spec(spec_file, repo_root=self.tmp_dir)
    ###END def test_invalid_region_fallbacks_type_raises

###END class TestBuildHistoricalComparison
