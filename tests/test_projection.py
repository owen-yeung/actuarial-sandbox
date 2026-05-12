from term_life.experience import mortality_ae_ratio, suggest_k_mort_from_ae
from term_life.projection import Assumptions, Portfolio, project_term_life
from term_life.tables import LapseCurve, MortalityTable


def test_no_decrements_flat_pv():
    table = MortalityTable({40: 0.0, 41: 0.0})
    lapse = LapseCurve({1: 0.0, 2: 0.0})
    p = Portfolio(1000, 40, 1000.0, 100.0, 0.0)
    a = Assumptions(2, table, lapse, k_mort=1.0, k_lapse=1.0, discount_rate=0.0)
    out = project_term_life(p, a)
    assert out.deaths == (0, 0)
    assert out.lapses == (0, 0)
    assert out.active_start[0] == 1000 and out.active_start[1] == 1000
    assert out.pv_premiums == 1000 * 100.0 * 2  # mid-year factor 1 when r=0


def test_all_lapse_year_one():
    table = MortalityTable({40: 0.0, 41: 0.0, 42: 0.0})
    lapse = LapseCurve({1: 1.0})
    p = Portfolio(100, 40, 1.0, 1.0, 0.0)
    a = Assumptions(3, table, lapse, k_mort=1.0, k_lapse=1.0, discount_rate=0.0)
    out = project_term_life(p, a)
    assert out.active_start[0] == 100
    assert out.lapses[0] == 100
    assert out.active_start[1] == 0


def test_mortality_ae_and_suggest_k():
    table = MortalityTable({50: 0.02})
    ae = mortality_ae_ratio(30.0, 1000.0, 50, table)
    assert ae is not None and abs(ae - 1.5) < 1e-9
    k = suggest_k_mort_from_ae(ae, partial_credibility=1.0)
    assert abs(k - 1.5) < 1e-9
