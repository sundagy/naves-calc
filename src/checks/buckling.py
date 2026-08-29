from __future__ import annotations

from math import pi


def euler_capacity_n(E: float, I_min: float, effective_length: float, k: float = 1.0) -> float:
    return pi**2 * E * I_min / (k * effective_length) ** 2

