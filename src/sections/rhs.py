from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RHSSection:
    name: str
    width: float
    height: float
    thickness: float
    rho: float = 7850.0

    def __post_init__(self) -> None:
        if not all(isfinite(v) and v > 0 for v in (self.width, self.height, self.thickness, self.rho)):
            raise ValueError("Section dimensions, thickness and density must be finite and positive")
        if 2 * self.thickness >= min(self.width, self.height):
            raise ValueError("Tube thickness leaves no hollow core")

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
        # Thin-walled closed section: J ≈ 4 A_m² t / p_m.
        # SCI P385, Appendix B.2.3. Square corners, uniform wall thickness;
        # this is an approximation, not a manufacturer's section-table value.
        mean_width = self.width - self.thickness
        mean_height = self.height - self.thickness
        mean_area = mean_width * mean_height
        mean_perimeter = 2 * (mean_width + mean_height)
        return 4 * mean_area**2 * self.thickness / mean_perimeter


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
