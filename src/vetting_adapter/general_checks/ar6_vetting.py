"""Vetting criteria and targets/ranges for IPCC AR6.

The full list of `CriterionTargetRange` objects for each of the AR6 vetting
criteria is given in the `vetting_targets` variable. Objects for the historical
and future criteria alone are given in the `vetting_targets_historical` and
`vetting_targets_future` variables, respectively.

Credit: The Criterion objects used here are originally taken from example
notebooks in the `pathways_ensemble_analysis` package repository.
"""
import typing as tp

import pandas as pd
import numpy as np

from pathways_ensemble_analysis.criteria.base import (
    Criterion,
    SingleVariableCriterion,
    ChangeOverTimeCriterion,
)

from .target_classes import (
    CriterionTargetRange,
    RelativeRange,
)



# Historical emissions. The ranges given are the widest ranges given in Table 11
# of AR6 WGIII Annex III
# (https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Annex-III.pdf#page=43).
# The vetting ranges for Illustrative Pathways is given in comments starting
# with `# IP range:`.

vetting_targets_historical: list[CriterionTargetRange] = [

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name='CO2 total (EIP + AFOLU) emissions 2020',
            region='World',
            year=2020,
            variable='Emissions|CO2',
            unit='Mt CO2 / yr',
        ),
        target=44251.0,
        unit='Mt CO2 / yr',
        range=RelativeRange(0.6, 1.4),  # IP range: +/- 20%
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name='CO2 EIP emissions 2020',
            region='World',
            year=2020,
            variable='Emissions|CO2|Energy and Industrial Processes',
            unit='Mt CO2 / yr',
        ),
        target=37646.0,
        unit='Mt CO2 / yr',
        range=RelativeRange(0.8, 1.2),  # IP range: +/- 10%
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name='CH4 emissions 2020',
            region='World',
            year=2020,
            variable='Emissions|CH4',
            unit='Mt CH4 / yr',
        ),
        target=379.0,
        unit='Mt CH4 / yr',
        range=RelativeRange(0.8, 1.2),  # IP range: +/- 20%
    ),

    CriterionTargetRange(
        criterion=ChangeOverTimeCriterion(
            criterion_name='CO2 EIP emissions 2010-2020 change',
            region='World',
            year=2020,
            reference_year=2010,
            variable='Emissions|CO2|Energy and Industrial Processes',
        ),
        target=0.25,
        range=(0.0, 0.5),
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name='CCS from energy 2020',
            region='World',
            year=2020,
            variable='Carbon Sequestration|CCS',   # Should be energy-only, but hard to find a variable that captures that consistently across all models. Total CCS in any case gives an upper bound, so scenarios that satisfy this criterion will also satisfy the actual AR6 criterion.
            unit='Mt CO2 / yr',
        ),
        target=0.0,
        unit='Mt CO2 / yr',
        range=(0.0, 250.0),  # IP range: 0-100 Mt CO2 / yr
        distance_func=lambda x: x/250.0,  # Use this to avoid division by zero in the defult function.
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name="Primary Energy 2020",
            region="World",
            year=2020,
            variable="Primary Energy",
            unit="EJ / yr",
        ),
        target=578.0,
        unit='EJ / yr',
        range=RelativeRange(0.8, 1.2),  # IP range: +/- 10%
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name="Electricity: nuclear 2020",
            region="World",
            year=2020,
            variable="Secondary Energy|Electricity|Nuclear",
            unit="EJ / yr",
        ),
        target=9.77,
        unit='EJ / yr',
        range=RelativeRange(0.7, 1.3),  # IP range: +/- 20%
    ),

    CriterionTargetRange(
        SingleVariableCriterion(
            criterion_name="Electricity: solar and wind 2020",
            region="World",
            year=2020,
            variable="Secondary Energy|Electricity|Wind and Solar",
            unit="EJ / yr",
        ),
        target=8.51,
        unit='EJ / yr',
        range=RelativeRange(0.5, 1.5),  # IP range: +/- 25%
    ),

]

vetting_targets_future: list[CriterionTargetRange] = [

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name="CO2 EIP emissions 2030",
            region="World",
            year=2030,
            variable="Emissions|CO2|Energy and Industrial Processes",
            unit="Mt CO2 / yr",
        ),
        target=44251.0,
        unit='Mt CO2 / yr',
        range=(0.0, np.inf),  # The way the distance function is defined, -1 is
                            # 0 emissions in 2030 (the actual vetting bound),
                            # while +1 is infinite emissions.
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name="CCS from energy 2030",
            region="World",
            year=2030,
            variable="Carbon Sequestration|CCS",  # Same comment about the
                                                  # variable name here as for
                                                  # the criterion named
                                                  # 'CCS from energy 2020'
            unit="Mt CO2 / yr",
        ),
        target=0.0,
        unit='Mt CO2 / yr',
        range=(0.0, 2000.0)
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name="Electricity: nuclear 2030",
            region="World",
            year=2030,
            variable="Secondary Energy|Electricity|Nuclear",
            unit="EJ / yr",
        ),
        target=0.0,
        unit='EJ / yr',
        range=(0.0, 20.0),
    ),

    CriterionTargetRange(
        criterion=SingleVariableCriterion(
            criterion_name="CH4 emissions 2040",
            region="World",
            year=2040,
            variable="Emissions|CH4",
            unit="Mt CH4 / yr",
        ),
        target=100.0*np.sqrt(1000.0/100.0),
        unit='Mt CH4 / yr',
        range=(100.0, 1000.0),
    ),

]


vetting_targets: list[CriterionTargetRange] = vetting_targets_historical \
    + vetting_targets_future


# For reference, below are the Criterion objects and drop condition
# specifications used in the AR6 assessment notebook from the
# `pathways-ensemble-analysis` repository.

_pea_vetting_criteria: list[Criterion] = [

    # Historical emissions and CCS

    SingleVariableCriterion(
        criterion_name="CO2 total (EIP + AFOLU) emissions 2020",
        region="World",
        year=2020,
        variable="Emissions|CO2",
        unit="Mt CO2 / yr",
    ),

    SingleVariableCriterion(
        criterion_name="CO2 EIP emissions 2020",
        region="World",
        year=2020,
        variable="Emissions|CO2|Energy and Industrial Processes",
        unit="Mt CO2 / yr",
    ),

    SingleVariableCriterion(
        criterion_name="CH4 emissions 2020",
        region="World",
        year=2020,
        variable="Emissions|CH4",
        unit="Mt CH4 / yr",
    ),

    ChangeOverTimeCriterion(
        criterion_name="CO2 EIP emissions 2010-2020 change",
        region="World",
        year=2020,
        reference_year=2010,
        variable="Emissions|CO2|Energy and Industrial Processes",
    ),

    SingleVariableCriterion(
        criterion_name="CCS from energy 2020",
        region="World",
        year=2020,
        variable="Carbon Sequestration|CCS|Energy",
        unit="Mt CO2 / yr",
    ),

    # Historical energy production

    SingleVariableCriterion(
        criterion_name="Primary Energy 2020",
        region="World",
        year=2020,
        variable="Primary Energy",
        unit="EJ / yr",
    ),

    SingleVariableCriterion(
        criterion_name="Electricity: nuclear 2020",
        region="World",
        year=2020,
        variable="Secondary Energy|Electricity|Nuclear",
        unit="EJ / yr",
    ),

    SingleVariableCriterion(
        criterion_name="Electricity: solar and wind 2020",
        region="World",
        year=2020,
        variable="Secondary Energy|Electricity|Wind and Solar",
        unit="EJ / yr",
    ),

    # Future criteria

    SingleVariableCriterion(
        criterion_name="CO2 EIP emissions 2030",
        region="World",
        year=2030,
        variable="Emissions|CO2|Energy and Industrial Processes",
        unit="Mt CO2 / yr",
    ),

    SingleVariableCriterion(
        criterion_name="CCS from energy 2030",
        region="World",
        year=2030,
        variable="Carbon Sequestration|CCS|Energy",
        unit="Mt CO2 / yr",
    ),

    SingleVariableCriterion(
        criterion_name="Electricity: nuclear 2030",
        region="World",
        year=2030,
        variable="Secondary Energy|Electricity|Nuclear",
        unit="EJ / yr",
    ),

    SingleVariableCriterion(
        criterion_name="CH4 emissions 2040",
        region="World",
        year=2040,
        variable="Emissions|CH4",
        unit="Mt CH4 / yr",
    ),

    # Additionally defined criteria (which are not investigated in the AR6
    # vetting process)

    SingleVariableCriterion(
        criterion_name="CO2 EIP emissions 2050",
        region="World",
        year=2050,
        variable="Emissions|CO2|Energy and Industrial Processes",
        unit="Mt CO2 / yr",
    ),

    SingleVariableCriterion(
        criterion_name="CCS from energy 2050",
        region="World",
        year=2050,
        variable="Carbon Sequestration|CCS|Energy",
        unit="Mt CO2 / yr",
    ),

    SingleVariableCriterion(
        criterion_name="Electricity: nuclear 2050",
        region="World",
        year=2050,
        variable="Secondary Energy|Electricity|Nuclear",
        unit="EJ / yr",
    ),

    SingleVariableCriterion(
        criterion_name="Electricity: solar and wind 2050",
        region="World",
        year=2050,
        variable="Secondary Energy|Electricity|Wind and Solar",
        unit="EJ / yr",
    ),

]

_pea_drop_conditions_historical: dict[str, dict[str, tp.Any]] = {
    # Historical emissions
    "CO2 total (EIP + AFOLU) emissions 2020": {
        "mode": "outside",
        "value": [44251.0 * 0.6, 44251.0 * 1.4],
    },
    "CO2 EIP emissions 2020": {"mode": "outside", "value": [37646 * 0.8, 37646 * 1.2]},
    "CH4 emissions 2020": {"mode": "outside", "value": [379 * 0.8, 379 * 1.2]},
    "CO2 EIP emissions 2010-2020 change": {"mode": "outside", "value": [0, 0.5]},
    "CCS from energy 2020": {"mode": ">=", "value": 250},
    # Historical energy production
    "Primary Energy 2020": {"mode": "outside", "value": [578 * 0.8, 578 * 1.2]},
    "Electricity: nuclear 2020": {"mode": "outside", "value": [9.77 * 0.7, 9.77 * 1.3]},
    "Electricity: solar and wind 2020": {
        "mode": "outside",
        "value": [8.51 * 0.5, 8.51 * 1.5],
    },
}

_pea_drop_conditions_future: dict[str, dict[str, tp.Any]] = {
    "CO2 EIP emissions 2030": {"mode": "<=", "value": 0},
    "CCS from energy 2030": {"mode": ">=", "value": 2000},
    "Electricity: nuclear 2030": {"mode": ">=", "value": 20},
    "CH4 emissions 2040": {"mode": "outside", "value": [100,1000]},
}
