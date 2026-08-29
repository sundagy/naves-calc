from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SteelMaterial:
    grade: str = "C245"
    E: float = 2.0e11
    nu: float = 0.3
    rho: float = 7850.0
    resistance_pa: float = 245e6

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


DEFAULT_STEEL = SteelMaterial()

