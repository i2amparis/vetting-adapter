"""Tests for the pdhelpers module."""
import unittest
import typing as tp

import pandas as pd

from vetting_adapter.pdhelpers import replace_level_values


class TestReplaceLevelValues(unittest.TestCase):
    """Tests for the `replace_level_values` function."""

    @classmethod
    def get_multiindex_df(cls) -> pd.DataFrame:
        """Creates a DataFrame with a MultiIndex for testing."""
        data: dict[str, list[int]] = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
        df: pd.DataFrame = pd.DataFrame(data)
        df.set_index(['A', 'B'], inplace=True)
        return df


    def test_replace_level_values_mapping(self):
        df: pd.DataFrame = self.get_multiindex_df()

        # Define the mapping
        mapping: dict[int, int] = {1: 10, 2: 20, 3: 30}

        # Call the function under test
        result: pd.DataFrame = replace_level_values(df, mapping, leveln=0)

        # Check if the values in the first level of the MultiIndex are replaced correctly
        expected: pd.MultiIndex = pd.MultiIndex.from_tuples([(10, 5), (20, 6), (30, 7), (4, 8)], names=['A', 'B'])
        self.assertEqual(result.index.names, expected.names)
        self.assertEqual(list(result.index), list(expected))

    def test_replace_level_values_mapping_level_name(self):
        df: pd.DataFrame = self.get_multiindex_df()

        # Define the mapping with only two elements
        mapping: dict[int, int] = {7: 70, 5: 50}

        # Call the function under test
        result: pd.DataFrame = replace_level_values(df, mapping, level_name='B')

        # Check if the values in the second level of the MultiIndex are replaced correctly
        expected: pd.MultiIndex = pd.MultiIndex.from_tuples([(1, 50), (2, 6), (3, 70), (4, 8)], names=['A', 'B'])
        self.assertEqual(result.index.names, expected.names)
        self.assertEqual(list(result.index), list(expected))

    def test_replace_level_values_invalid_leveln(self):
        """IndexError is raised when the level does not exist."""
        df: pd.DataFrame = self.get_multiindex_df()

        # Define the mapping
        mapping: dict[int, int] = {1: 10, 2: 20, 3: 30}

        # Call the function under test and check if an error is raised
        with self.assertRaises(IndexError):
            replace_level_values(df, mapping, leveln=2)

        # Check if an error is raised when using an invalid name for the `level_name` parameter
        with self.assertRaises(ValueError):
           replace_level_values(df, mapping, level_name='C')

    def test_replace_level_values_no_multiindex(self):
        """TypeError is raised when the DataFrame does not have a MultiIndex."""
        df: pd.DataFrame = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})

        # Define the mapping
        mapping: dict[int, int] = {1: 10, 2: 20, 3: 30}

        # Call the function under test and check if an error is raised
        with self.assertRaises(TypeError):
            replace_level_values(df, mapping, leveln=0)

    def test_replace_level_values_mapping_duplicate_keys(self):
        df: pd.DataFrame = self.get_multiindex_df()

        # Define the mapping
        mapping: dict[int, int] = {1: 2, 2: 20, 3: 20, 4: 15}

        # Call the function under test
        result: pd.DataFrame = replace_level_values(df, mapping, leveln=0)

        # Check if the values in the first level of the MultiIndex are replaced correctly
        expected: pd.MultiIndex = pd.MultiIndex.from_tuples([(2, 5), (20, 6), (20, 7), (15, 8)], names=['A', 'B'])
        self.assertEqual(result.index.names, expected.names)
        self.assertEqual(list(result.index), list(expected))
