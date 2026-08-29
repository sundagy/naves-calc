from __future__ import annotations

import numpy as np

from src.config import load_normative
from src.geometry.canopy import RoofPlane


def mu_for_mono_slope(angle_deg: float) -> float:
    data = load_normative()["snow"]["mono_slope_mu"]
    points = data["points"]
    angles = [float(point[0]) for point in points]
    values = [float(point[1]) for point in points]
    # СП 20.13330.2016, приложение Б: коэффициент формы для односкатного покрытия
    return float(np.interp(max(angle_deg, 0.0), angles, values, left=values[0], right=values[-1]))


def snow_design_pressure_kpa(
    sg_kpa: float,
    ce: float,
    ct: float,
    roof: RoofPlane,
    gamma_f: float = 1.4,
    mu: float | None = None,
) -> tuple[float, float, float]:
    form_mu = mu_for_mono_slope(roof.max_angle_deg) if mu is None else mu
    s0 = ce * ct * form_mu * sg_kpa
    sd = gamma_f * s0
    return form_mu, s0, sd


def kaluga_source() -> str:
    return load_normative()["snow"]["kaluga"]["source"]

