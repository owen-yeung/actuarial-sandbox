import pandas as pd

from term_life.version_store import (
    VersionGraph,
    apply_payload_to_session,
    bundle_for_share,
    collect_payload_from_session,
    decode_bundle,
    encode_bundle,
    graph_from_bundle,
)


def test_encode_roundtrip():
    g = VersionGraph()
    g.add_snapshot(parent_id=None, name="root", notes="init", payload={"n0": 1000, "issue_age": 40})
    sid = g.head_id
    assert sid is not None
    b = bundle_for_share(g, sid)
    t = encode_bundle(b)
    b2 = decode_bundle(t)
    g2, focus = graph_from_bundle(b2)
    assert focus == sid
    assert len(g2.nodes) == 1
    assert g2.nodes[sid].name == "root"


def test_apply_payload_policy_roundtrip():
    st: dict = {}
    df = pd.DataFrame({"policy_id": ["a"], "issue_age": [40], "gender": ["M"], "smoker": ["N"], "sum_assured": [1.0], "policy_year": [1], "exposure_years": [1.0], "died": [0], "lapsed": [0]})
    st["policy_exp_df"] = df
    p = collect_payload_from_session(st)
    st2: dict = {}
    apply_payload_to_session(st2, p)
    assert st2["n0"] == 100_000
    assert st2["policy_exp_df"] is not None
    assert len(st2["policy_exp_df"]) == 1
