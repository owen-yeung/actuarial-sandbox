"""Limited-fluctuation style credibility helpers (illustrative, for sandbox)."""

from __future__ import annotations


def limited_fluctuation_z(num_claims: float, full_credibility_claims: float = 1082.0) -> float:
    """
    Classical limited-fluctuation Z = min(1, sqrt(n / n_full)) using claim counts.

    Matches the markdown example: 270 deaths vs 1082 full credibility -> Z ≈ 0.5.
    """
    if full_credibility_claims <= 0 or num_claims <= 0:
        return 0.0
    return min(1.0, (num_claims / full_credibility_claims) ** 0.5)


def credibility_weighted_ae(observed_ae: float, z: float, prior_ae: float = 1.0) -> float:
    """Blended A/E = Z * observed + (1 - Z) * prior (e.g. prior = 1.0)."""
    z = max(0.0, min(1.0, z))
    return z * observed_ae + (1.0 - z) * prior_ae
