import io

import pandas as pd
import pytest

from term_life.csv_loaders import lapse_curve_from_csv, mortality_table_from_csv
from term_life.tables import stylized_base_mortality_table


def test_mortality_csv_merge():
    csv = io.StringIO("age,q\n50,0.01\n")
    df = pd.read_csv(csv)
    t = mortality_table_from_csv(df, merge_with_stylized=True)
    assert t.q(50) == 0.01
    assert t.q(40) == stylized_base_mortality_table().q(40)


def test_lapse_csv_merge():
    csv = io.StringIO("policy_year,lapse\n1,0.1\n")
    df = pd.read_csv(csv)
    c = lapse_curve_from_csv(df, term_years=5, merge_with_stylized=True)
    assert c.rate(1) == 0.1
    assert c.rate(3) == pytest.approx(0.03, rel=1e-6)


def test_mortality_csv_bad_column():
    df = pd.read_csv(io.StringIO("foo,bar\n1,2\n"))
    with pytest.raises(ValueError):
        mortality_table_from_csv(df)
