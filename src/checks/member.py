"""Elastic stress and Euler screening; neither is a complete SP16 check."""
from math import isfinite

from src.checks.buckling import euler_capacity_n


def member_screening(section, steel, axial_n, local_force, effective_length=None):
    # OpenSees 3D localForce: N,Vy,Vz,T,My,Mz at i, then at j.
    # Section local y = depth, local z = width (see beam_width_axis).
    my = max(abs(local_force[4]), abs(local_force[10])) if local_force else 0.0
    mz = max(abs(local_force[5]), abs(local_force[11])) if local_force else 0.0
    stress = abs(axial_n) / section.area
    if local_force:
        stress = max(abs(local_force[k]) / section.area
                     + abs(local_force[k + 4]) / section.wy
                     + abs(local_force[k + 5]) / section.wx for k in (0, 6))
    compression = max(0.0, -axial_n)
    capacity = ratio = None
    if effective_length is not None:
        if not isfinite(effective_length) or effective_length <= 0:
            raise ValueError("Расчётная длина должна быть конечной и положительной")
        capacity = euler_capacity_n(steel.E, min(section.ix, section.iy), effective_length)
        ratio = compression / capacity
    return {
        "My_max_kNm": my / 1000, "Mz_max_kNm": mz / 1000,
        "elastic_NM_util": stress / steel.resistance_pa,
        "effective_length_m": effective_length,
        "Euler_Ncr_kN": capacity / 1000 if capacity is not None else None,
        "Euler_ratio": ratio,
    }
