import io

import pandas as pd
import pytest

from term_life.credibility import credibility_weighted_ae, limited_fluctuation_z
from term_life.policy_experience import (
    expected_decrements_from_rollups,
    parse_policy_level_experience,
    rollup_lapse_by_policy_year,
    rollup_mortality_by_age,
)
from term_life.tables import stylized_base_lapse_by_duration, stylized_base_mortality_table


def test_limited_fluctuation_z_example():
    assert limited_fluctuation_z(270, 1082) == pytest.approx((270 / 1082) ** 0.5)


def test_credibility_weighted_ae():
    assert credibility_weighted_ae(1.15, 0.5, 1.0) == pytest.approx(1.075)


def test_policy_parse_and_rollup():
    csv = """policy_id,issue_age,gender,smoker,sum_assured,policy_year,exposure_years,died,lapsed
P1,40,M,N,100000,1,1.0,1,0
P2,40,F,Y,250000,1,1.0,0,0
P2,40,F,Y,250000,2,1.0,0,1
"""
    df = pd.read_csv(io.StringIO(csv))
    p = parse_policy_level_experience(df)
    m = rollup_mortality_by_age(p)
    assert set(m["attained_age"]) == {40, 41}
    assert m["exposure_years"].sum() == 3.0
    assert m["observed_deaths"].sum() == 1.0
    l = rollup_lapse_by_policy_year(p)
    assert l["observed_lapses"].sum() == 1.0


def test_expected_decrements():
    mort = stylized_base_mortality_table()
    lapse = stylized_base_lapse_by_duration(20)
    csv = """policy_id,issue_age,gender,smoker,sum_assured,policy_year,exposure_years,died,lapsed
P1,40,M,N,100000,1,1.0,0,0
"""
    p = parse_policy_level_experience(pd.read_csv(io.StringIO(csv)))
    m = rollup_mortality_by_age(p)
    lp = rollup_lapse_by_policy_year(p)
    em, el = expected_decrements_from_rollups(m, lp, mort, lapse)
    q40 = mort.q(40)
    assert em["expected_deaths"].iloc[0] == pytest.approx(1.0 * q40)
    assert el["expected_lapses"].iloc[0] == pytest.approx(1.0 * lapse.rate(1))
