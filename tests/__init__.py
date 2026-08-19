"""Common resources for testing"""
import typing as tp

import pyam
import pandas as pd
import numpy as np

from vetting_adapter.pdhelpers import replace_level_values



TV = tp.TypeVar('TV')
def notnone(x: TV|None) -> TV:
    assert x is not None
    return x
###END def notnone


def get_test_energy_iamdf_tuple() -> tuple[
        pyam.IamDataFrame,
        pyam.IamDataFrame,
        pyam.IamDataFrame,
        pyam.IamDataFrame,
]:
    """Make IamDataFrames for testing comparisons using TimeseriesRefCriterion.
    
    Returns
    -------
    data_df, target_df, diff_df, ratio_df : (IamDataFrame, IamDataFrame)
        A tuple of four IamDataFrames, the first being the test IamDataFrame and
        the second being the target IamDataFrame. They each have the following
        structure:
          - data_models: ['ModelA', 'ModelB'] for `data_df` and ['Target Model']
            for `target_df`
          - data_scenarios: ['Scenario1', 'Scenario2']
          - regions: ['Region1', 'Region2', 'Region3']
          - years: [2005, 2010, 2015, 2020, 2025, 2030]
          - variables: ['Primary Energy', 'Secondary Energy|Electricity']
          - units: ['EJ/yr', 'TWh/yr']
        The values for 'Primary Energy' for both models and
        'Secondary Energy|Electricity' for 'ModelA' have units `EJ/yr` and vary
        randomly between 1 and 100. The values for
        'Secondary Energy|Electricity' for 'ModelB' have units `TWh/yr` and vary
        randomly 1000/3.6 and 100000/3.6 (the same range as the `EJ/yr` values,
        but converted to `TWh/yr`).
        The last two IamDataFrames have the same structure as `data_df`, and
        values equal to the difference and ratio of the values in `data_df`
        relative to `target_df`, respectively (matching on all dimensions
        except for the `model` and `unit` dimensions). They are both in units
        of `EJ/yr`.
    """
    data_models: list[str] = ['ModelA', 'ModelB']
    data_scenarios: list[str] = ['Scenario1', 'Scenario2']
    regions: list[str] = ['Region1', 'Region2', 'Region3']
    years: list[int] = [2005, 2010, 2015, 2020, 2025, 2030]
    variables: list[str] = ['Primary Energy', 'Secondary Energy|Electricity']
    index_without_units: pd.MultiIndex = pd.MultiIndex.from_product(
        [data_models, data_scenarios, regions, variables, years],
        names=['model', 'scenario', 'region', 'variable', 'year']
    )
    units_arrays: list[str] = [
        'TWh/yr' if _model == 'ModelB' and _variable == 'Secondary Energy|Electricity'
        else 'EJ/yr'
        for _model, _variable in zip(index_without_units.get_level_values('model'),
                                     index_without_units.get_level_values('variable'))
    ]
    index: pd.MultiIndex = pd.DataFrame(  # type: ignore
        data=units_arrays,
        index=index_without_units,
        columns=['unit']
    ).set_index('unit', append=True).index
    data_series: pd.Series = pd.Series(
        data=np.random.rand(len(index)) * 99.0 + 1.0,
        index=index
    )
    # Create a target series that is numerically equal to data_series for
    # `model == "ModelB"`, but with the model name replaced by `"Target Model"`
    target_series: pd.Series = data_series.xs('ModelB', level='model',
                                              drop_level=False)
    target_series = replace_level_values(
        target_series,
        mapping={'ModelB': 'Target Model'},
        level_name='model'
    )
    target_series_allEJ = replace_level_values(
        target_series,
        mapping={'TWh/yr': 'EJ/yr'},
        level_name='unit'
    )
    target_series_electTWh = target_series.where(
        target_series.index.get_level_values('unit') != 'TWh/yr',
        other=target_series * 1000.0 / 3.6
    )

    # Multiply the target_series by 1000 and divide by 3.6 where the unit is
    # 'TWh/yr' and the model is 'ModelB' and the variable is 'Secondary
    # Energy|Electricity'.
    data_series = data_series.where(
        (data_series.index.get_level_values('unit') == 'EJ/yr') |
        (data_series.index.get_level_values('model') == 'ModelA') |
        (data_series.index.get_level_values('variable') == 'Primary Energy'),
        other=data_series * 1000.0 / 3.6
    )

    diff_series: pd.Series = tp.cast(pd.Series, pd.concat(
        [
            data_series.xs('ModelA', level='model', drop_level=False) \
                - replace_level_values(
                    target_series_allEJ,
                    mapping={'Target Model': 'ModelA'},
                    level_name='model'
                ),
            data_series.xs('ModelB', level='model', drop_level=False) \
                - replace_level_values(
                    target_series_electTWh,
                    mapping={'Target Model': 'ModelB'},
                    level_name='model'
                )
        ]
    ))
    ratio_series: pd.Series = pd.concat(
        [
            data_series.xs('ModelA', level='model', drop_level=False) \
                / replace_level_values(
                    target_series_allEJ,
                    mapping={'Target Model': 'ModelA'},
                    level_name='model'
                ),
            data_series.xs('ModelB', level='model', drop_level=False) \
                / replace_level_values(
                    target_series_electTWh,
                    mapping={'Target Model': 'ModelB'},
                    level_name='model'
                )
        ]
    )
    ratio_series = replace_level_values(
        ratio_series,
        mapping={_unit: '' for _unit in ['EJ/yr', 'TWh/yr']},
        level_name='unit'
    )
    # The units of the target_series should be 'EJ/yr' everywhere, and not
    # converted, so set the `unit` level of the target series to 'EJ/yr'
    # everywhere.
    assert isinstance(target_series_electTWh.index, pd.MultiIndex)
    # target_series.index = target_series.index.set_levels(['EJ/yr'], level='unit')
    # values: pd.Series = pd.Series(
    #     data=[1.0 + 99.0 * i / 5.0 for i in range(6 * 2 * 3 * 2 * 2)],
    #     index=index
    # )
    return (
        pyam.IamDataFrame(data_series),
        notnone(pyam.IamDataFrame(target_series_electTWh)),
        notnone(pyam.IamDataFrame(diff_series)),
        notnone(pyam.IamDataFrame(ratio_series))
    )
###END def construct_test_iamdf
