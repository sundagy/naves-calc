from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RHSSection:
    name: str
    width: float
    height: float
    thickness: float
    rho: float = 7850.0

    @property
    def area(self) -> float:
        inner_width = self.width - 2.0 * self.thickness
        inner_height = self.height - 2.0 * self.thickness
        return self.width * self.height - inner_width * inner_height

    @property
    def ix(self) -> float:
        inner_width = self.width - 2.0 * self.thickness
        inner_height = self.height - 2.0 * self.thickness
        return (self.width * self.height**3 - inner_width * inner_height**3) / 12.0

    @property
    def iy(self) -> float:
        inner_width = self.width - 2.0 * self.thickness
        inner_height = self.height - 2.0 * self.thickness
        return (self.height * self.width**3 - inner_height * inner_width**3) / 12.0

    @property
    def wx(self) -> float:
        return self.ix / (self.height / 2.0)

    @property
    def wy(self) -> float:
        return self.iy / (self.width / 2.0)

    @property
    def rx(self) -> float:
        return (self.ix / self.area) ** 0.5

    @property
    def ry(self) -> float:
        return (self.iy / self.area) ** 0.5

    @property
    def mass_per_m(self) -> float:
        return self.area * self.rho

    @property
    def torsion_j(self) -> float:
        return self.ix + self.iy


def parse_rhs(name: str) -> RHSSection:
    parts = name.lower().replace("rhs", "").replace("shs", "").split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid RHS/SHS section name: {name}")
    width_mm, height_mm, thickness_mm = (float(part) for part in parts)
    if thickness_mm * 2 >= min(width_mm, height_mm):
        raise ValueError(f"Invalid tube thickness for section: {name}")
    return RHSSection(
        name=name,
        width=width_mm / 1000.0,
        height=height_mm / 1000.0,
        thickness=thickness_mm / 1000.0,
    )

