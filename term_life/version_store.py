"""
In-memory version graph for the Streamlit sandbox: snapshots, notes, serialize/share.

No server persistence — state lives in ``st.session_state`` and optional URL query params.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

FORMAT_TAG = "actuarial-sandbox-vc-1"
# Conservative limit for query-string sharing on localhost.
MAX_SHARE_URL_CHARS = 4500


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Snapshot:
    id: str
    parent_id: str | None
    name: str
    notes: str
    created_at: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "name": self.name,
            "notes": self.notes,
            "created_at": self.created_at,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Snapshot:
        return cls(
            id=str(d["id"]),
            parent_id=d.get("parent_id"),
            name=str(d["name"]),
            notes=str(d.get("notes", "")),
            created_at=str(d["created_at"]),
            payload=dict(d["payload"]),
        )


@dataclass
class VersionGraph:
    """Directed graph of snapshots (parent → child)."""

    nodes: dict[str, Snapshot] = field(default_factory=dict)
    """When set, working UI was last aligned to this snapshot (lineage anchor for new saves)."""

    head_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "head_id": self.head_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VersionGraph:
        nodes = {k: Snapshot.from_dict(v) for k, v in d.get("nodes", {}).items()}
        return cls(nodes=nodes, head_id=d.get("head_id"))

    def add_snapshot(
        self,
        *,
        parent_id: str | None,
        name: str,
        notes: str,
        payload: dict[str, Any],
    ) -> str:
        sid = str(uuid.uuid4())[:12]
        self.nodes[sid] = Snapshot(
            id=sid,
            parent_id=parent_id,
            name=name.strip() or "Untitled",
            notes=notes.strip(),
            created_at=_iso_now(),
            payload=payload,
        )
        self.head_id = sid
        return sid

    def dot_graph(self, *, highlight_id: str | None = None) -> str:
        """Graphviz DOT for st.graphviz_chart."""
        lines = ["digraph G {", '  rankdir="LR";', "  node [shape=box, style=rounded];"]
        for sid, sn in self.nodes.items():
            label = f"{sn.name}\\n{sn.id}"
            if len(sn.notes) > 40:
                label += "\\n" + sn.notes[:37] + "…"
            elif sn.notes:
                label += "\\n" + sn.notes.replace('"', "'")
            esc = label.replace("\\", "\\\\").replace('"', '\\"')
            color = "#2E86AB" if sid == highlight_id else "#333333"
            lines.append(f'  "{sid}" [label="{esc}", fontcolor="{color}"];')
        for sid, sn in self.nodes.items():
            if sn.parent_id and sn.parent_id in self.nodes:
                lines.append(f'  "{sn.parent_id}" -> "{sid}";')
        lines.append("}")
        return "\n".join(lines)


def collect_payload_from_session(st: Any) -> dict[str, Any]:
    """Build a JSON-serializable payload from Streamlit session_state-like mapping."""
    policy_csv: str | None = None
    pdf = st.get("policy_exp_df")
    if pdf is not None and hasattr(pdf, "to_csv"):
        policy_csv = pdf.to_csv(index=False)

    return {
        "n0": int(st.get("n0", 100_000)),
        "issue_age": int(st.get("issue_age", 40)),
        "term_years": int(st.get("term_years", 20)),
        "sa": float(st.get("sa", 100_000.0)),
        "prem": float(st.get("prem", 500.0)),
        "exp": float(st.get("exp", 50.0)),
        "disc": float(st.get("disc", 0.03)),
        "k_mort": float(st.get("k_mort", 0.95)),
        "k_lapse": float(st.get("k_lapse", 1.0)),
        "mort_mode": st.get("mort_mode", "Built-in stylized"),
        "lapse_mode": st.get("lapse_mode", "Built-in stylized"),
        "policy_exp_csv": policy_csv,
        "mort_csv_text": st.get("vc_mort_csv_text"),
        "lapse_csv_text": st.get("vc_lapse_csv_text"),
        "exp_k_mort": st.get("exp_k_mort"),
        "exp_k_lapse": st.get("exp_k_lapse"),
        "n_full_deaths": float(st.get("n_full_deaths", 1082.0)),
        "n_full_lapses": float(st.get("n_full_lapses", 1082.0)),
        "wf_step": int(st.get("wf_step", 1)),
        "cred_w": float(st.get("cred_w", 1.0)),
    }


def apply_payload_to_session(st: Any, payload: Mapping[str, Any]) -> None:
    """Write snapshot payload into session_state (widget keys + experience + embedded CSV)."""
    st["n0"] = int(payload.get("n0", 100_000))
    st["issue_age"] = int(payload.get("issue_age", 40))
    st["term_years"] = int(payload.get("term_years", 20))
    st["sa"] = float(payload.get("sa", 100_000.0))
    st["prem"] = float(payload.get("prem", 500.0))
    st["exp"] = float(payload.get("exp", 50.0))
    st["disc"] = float(payload.get("disc", 0.03))
    st["k_mort"] = float(payload.get("k_mort", 0.95))
    st["k_lapse"] = float(payload.get("k_lapse", 1.0))
    st["mort_mode"] = payload.get("mort_mode", "Built-in stylized")
    st["lapse_mode"] = payload.get("lapse_mode", "Built-in stylized")
    st["cred_w"] = float(payload.get("cred_w", 1.0))
    st["n_full_deaths"] = float(payload.get("n_full_deaths", 1082.0))
    st["n_full_lapses"] = float(payload.get("n_full_lapses", 1082.0))
    st["wf_step"] = int(payload.get("wf_step", 1))
    st["wf_step_radio"] = int(payload.get("wf_step", 1))

    exp_m = payload.get("exp_k_mort")
    exp_l = payload.get("exp_k_lapse")
    st["exp_k_mort"] = float(exp_m) if exp_m is not None else None
    st["exp_k_lapse"] = float(exp_l) if exp_l is not None else None

    pc = payload.get("policy_exp_csv")
    if pc:
        import pandas as pd

        st["policy_exp_df"] = pd.read_csv(io.StringIO(pc))
    else:
        st["policy_exp_df"] = None

    mtxt = payload.get("mort_csv_text")
    ltxt = payload.get("lapse_csv_text")
    st["vc_mort_csv_text"] = mtxt if mtxt else None
    st["vc_lapse_csv_text"] = ltxt if ltxt else None


def bundle_for_share(graph: VersionGraph, focus_id: str) -> dict[str, Any]:
    if focus_id not in graph.nodes:
        raise KeyError(f"Unknown snapshot id: {focus_id}")
    return {
        "format": FORMAT_TAG,
        "graph": graph.to_dict(),
        "focus_id": focus_id,
    }


def encode_bundle(bundle: Mapping[str, Any]) -> str:
    raw = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode("ascii")


def decode_bundle(token: str) -> dict[str, Any]:
    raw = zlib.decompress(base64.urlsafe_b64decode(token.encode("ascii")))
    data = json.loads(raw.decode("utf-8"))
    if data.get("format") != FORMAT_TAG:
        raise ValueError("Unrecognized share bundle format.")
    return data


def share_token_fits_url(token: str) -> bool:
    return len(token) <= MAX_SHARE_URL_CHARS


def graph_from_bundle(bundle: Mapping[str, Any]) -> tuple[VersionGraph, str]:
    g = VersionGraph.from_dict(bundle["graph"])
    focus = str(bundle["focus_id"])
    return g, focus
