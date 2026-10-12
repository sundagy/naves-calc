from __future__ import annotations

from dataclasses import dataclass, field
from math import dist, isfinite

import pandas as pd

from src.checks.member import member_screening
from src.checks.readiness import assessment
from src.loads.scenarios import snow_pattern_loads
from src.geometry.canopy import roof_plane_from_input
from src.fem.assumptions import beam_width_axis, node_restraints, uses_beam, apply_bearing_constraints
from src.loads.self_weight import G_ACCEL, mass_breakdown, self_weight_nodal_loads
from src.loads.snow_sp20 import snow_design_pressure_kpa
from src.loads.tributary import tributary_roof_loads
from src.materials.steel import SteelMaterial
from src.models import CanopyInput, LoadBreakdown, StructuralModel
from src.sections.database import section_database


@dataclass
class FEMResult:
    ok: bool
    nodes: pd.DataFrame
    elements: pd.DataFrame
    reactions: pd.DataFrame
    loads: pd.DataFrame
    load_breakdown: LoadBreakdown
    max_uz_mm: float
    max_abs_n_kn: float
    total_reaction_z_n: float
    message: str = ""
    artificial_reactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenarios: pd.DataFrame = field(default_factory=pd.DataFrame)
    equilibrium: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenario: str = "UNIFORM"
    reliability_confirmed: bool = False


VERTICAL_BALANCE_TOLERANCE_KN = 1e-6


def _load_breakdown(data, model, steel, mu, s0_kpa, sd_kpa, snow_loads, dead_loads, self_loads, nodal_loads, reaction_n):
    masses = mass_breakdown(model, steel)
    snow_kn = -sum(snow_loads.values()) / 1000.0
    dead_kn = -sum(dead_loads.values()) / 1000.0
    self_kn = -sum(self_loads.values()) / 1000.0
    applied_kn = -sum(nodal_loads.values()) / 1000.0
    lower_kn = masses["lower"] * G_ACCEL / 1000.0
    upper_kn = masses["upper"] * G_ACCEL / 1000.0
    columns_kn = masses["columns"] * G_ACCEL / 1000.0
    crossbeams_kn = masses["crossbeams"] * G_ACCEL / 1000.0
    components_kn = snow_kn + dead_kn + lower_kn + upper_kn + columns_kn + crossbeams_kn
    if abs(self_kn - (lower_kn + upper_kn + columns_kn + crossbeams_kn)) >= VERTICAL_BALANCE_TOLERANCE_KN:
        raise ValueError("Structural self-weight groups do not sum to total self-weight")
    if abs(applied_kn - (snow_kn + dead_kn + self_kn)) >= VERTICAL_BALANCE_TOLERANCE_KN:
        raise ValueError("Snow, roof dead load and structural self-weight do not sum to applied load")
    if abs(applied_kn - components_kn) >= VERTICAL_BALANCE_TOLERANCE_KN:
        raise ValueError(f"Vertical load components do not sum to applied load: {components_kn} vs {applied_kn} kN")
    return LoadBreakdown(
        sg_kpa=data.snow_sg_kpa, ce=data.snow_ce, ct=data.snow_ct,
        mu=mu, gamma_f=data.snow_gamma_f, s0_kpa=s0_kpa, sd_kpa=sd_kpa,
        dead_load_kpa=data.roof_dead_load_kpa,
        total_roof_load_n=(snow_kn + dead_kn) * 1000.0,
        total_self_weight_n=self_kn * 1000.0,
        snow_load_total_kN=snow_kn,
        roof_dead_load_total_kN=dead_kn,
        lower_trusses_self_weight_kN=lower_kn,
        upper_trusses_self_weight_kN=upper_kn,
        columns_self_weight_kN=columns_kn,
        crossbeams_self_weight_kN=crossbeams_kn,
        total_self_weight_kN=self_kn,
        total_applied_vertical_load_kN=applied_kn,
        sum_vertical_reactions_kN=reaction_n / 1000.0,
    )


def solve_model(data: CanopyInput, model: StructuralModel) -> FEMResult:
    try:
        import openseespy.opensees as ops
    except ImportError as exc:
        raise RuntimeError("OpenSeesPy is required. Run `uv sync`.") from exc

    steel = SteelMaterial(grade=data.steel_grade, resistance_pa=data.steel_resistance_mpa * 1e6)
    sections = section_database()
    roof = roof_plane_from_input(data)
    mu, s0_kpa, sd_kpa = snow_design_pressure_kpa(
        data.snow_sg_kpa,
        data.snow_ce,
        data.snow_ct,
        roof,
        data.snow_gamma_f,
        data.snow_mu,
    )
    snow_loads = snow_pattern_loads(data, model, sd_kpa)
    dead_loads = tributary_roof_loads(data, model, data.roof_dead_load_kpa)
    self_loads = self_weight_nodal_loads(model, steel)
    nodal_loads: dict[int, float] = {}
    for source in (snow_loads, dead_loads, self_loads):
        for node_id, fz in source.items():
            nodal_loads[node_id] = nodal_loads.get(node_id, 0.0) + fz

    ops.wipe()
    ops.model("BasicBuilder", "-ndm", 3, "-ndf", 6)
    for node in model.nodes.values():
        ops.node(node.id, node.x, node.y, node.z)
    restraints = node_restraints(data, model)
    for node_id, fixed in restraints.items():
        ops.fix(node_id, *fixed)

    ops.uniaxialMaterial("Elastic", 1, steel.E)
    apply_bearing_constraints(ops, data, model)
    for element in model.elements:
        section = sections[element.section]
        if uses_beam(data, element):
            transform = element.id
            ops.geomTransf("Linear", transform, *beam_width_axis(model, element))
            ops.element(
                "elasticBeamColumn",
                element.id,
                element.i,
                element.j,
                section.area,
                steel.E,
                steel.G,
                section.torsion_j,
                section.iy,
                section.ix,
                transform,
            )
        else:
            ops.element("truss", element.id, element.i, element.j, section.area, 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for node_id, fz in nodal_loads.items():
        ops.load(node_id, 0.0, 0.0, fz, 0.0, 0.0, 0.0)

    ops.constraints("Transformation" if model.bearing_links else "Plain")
    ops.numberer("RCM")
    ops.system("UmfPack" if model.mesh_node_ids or data.spatial_frame else "BandGeneral")
    ops.test("NormDispIncr", 1e-12, 20)
    # Correct the residual after the first solve for closely spaced bearing nodes.
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    code = ops.analyze(1)
    if code != 0:
        return _failed_result(data, model, steel, snow_loads, dead_loads, self_loads, nodal_loads, mu, s0_kpa, sd_kpa, code)

    ops.reactions()
    artificial_rows = []
    reaction_columns = ("Rx_kN", "Ry_kN", "Rz_kN", "Mx_kNm", "My_kNm", "Mz_kNm")
    for node_id, fixed in restraints.items():
        if node_id in model.support_node_ids:
            continue
        reaction = ops.nodeReaction(node_id)
        artificial_rows.append({
            "node": node_id, "label": model.nodes[node_id].label,
            "fixed_dofs": ", ".join(name for name, flag in zip(("Ux", "Uy", "Uz", "Rx", "Ry", "Rz"), fixed) if flag),
            **{name: float(reaction[k]) / 1000 if fixed[k] else 0.0
               for k, name in enumerate(reaction_columns)},
        })
    displacement_rows = []
    displacements: dict[int, tuple[float, float, float]] = {}
    for node in model.nodes.values():
        disp = ops.nodeDisp(node.id)
        if not all(isfinite(v) for v in disp):
            raise ValueError("Non-finite FEM displacements: result is invalid")
        ux, uy, uz = float(disp[0]), float(disp[1]), float(disp[2])
        displacements[node.id] = (ux, uy, uz)
        displacement_rows.append(
            {
                "node": node.id,
                "label": node.label,
                "x": node.x,
                "y": node.y,
                "z": node.z,
                "Ux_mm": ux * 1000.0,
                "Uy_mm": uy * 1000.0,
                "Uz_mm": uz * 1000.0,
                "roles": ", ".join(sorted(node.roles)),
            }
        )

    reaction_rows = []
    total_reaction_z = 0.0
    for column in model.columns:
        reaction = ops.nodeReaction(column.base_node)
        rz = float(reaction[2])
        total_reaction_z += rz
        reaction_rows.append(
            {
                "column": column.name,
                "base_node": column.base_node,
                "Rx_kN": float(reaction[0]) / 1000.0,
                "Ry_kN": float(reaction[1]) / 1000.0,
                "Rz_kN": rz / 1000.0,
                "Mx_kNm": float(reaction[3]) / 1000.0,
                "My_kNm": float(reaction[4]) / 1000.0,
                "Mz_kNm": float(reaction[5]) / 1000.0,
            }
        )

    breakdown = _load_breakdown(
        data, model, steel, mu, s0_kpa, sd_kpa,
        snow_loads, dead_loads, self_loads, nodal_loads, total_reaction_z,
    )
    if abs(breakdown.total_applied_vertical_load_kN - breakdown.sum_vertical_reactions_kN) >= VERTICAL_BALANCE_TOLERANCE_KN:
        raise ValueError(
            "Vertical reactions do not balance applied load: "
            f"{breakdown.sum_vertical_reactions_kN} vs {breakdown.total_applied_vertical_load_kN} kN"
        )

    element_rows = []
    for element in model.elements:
        ni = model.nodes[element.i]
        nj = model.nodes[element.j]
        length = dist(ni.coords, nj.coords)
        direction = ((nj.x - ni.x) / length, (nj.y - ni.y) / length, (nj.z - ni.z) / length)
        ui = displacements[element.i]
        uj = displacements[element.j]
        axial_extension = sum((uj[k] - ui[k]) * direction[k] for k in range(3))
        section = sections[element.section]
        axial_n = steel.E * section.area / length * axial_extension
        local_force = None
        if uses_beam(data, element):
            local_force = list(ops.eleResponse(element.id, "localForce"))
            if len(local_force) != 12:
                raise RuntimeError("OpenSees did not return 12 local beam forces")
            axial_n = (local_force[6] - local_force[0]) / 2
        screening = member_screening(section, steel, axial_n, local_force,
                                     data.effective_lengths_m.get(element.label))
        stress_pa = axial_n / section.area
        utilization = abs(stress_pa) / steel.resistance_pa
        element_rows.append(
            {
                "element": element.id,
                "i": element.i,
                "j": element.j,
                "role": element.role,
                "formulation": "beam" if uses_beam(data, element) else "truss",
                "group": element.group,
                "label": element.label,
                "section": element.section,
                "length_m": length,
                "mass_kg": section.area * length * steel.rho,
                "N_kN": axial_n / 1000.0,
                "stress_MPa": stress_pa / 1e6,
                "utilization": utilization,
                **screening,
            }
        )

    loads_rows = [{"node": node_id, "Fz_kN": fz / 1000.0} for node_id, fz in sorted(nodal_loads.items())]
    max_uz_mm = min(row["Uz_mm"] for row in displacement_rows)
    max_abs_n_kn = max(abs(row["N_kN"]) for row in element_rows) if element_rows else 0.0
    result = FEMResult(
        ok=True,
        nodes=pd.DataFrame(displacement_rows),
        elements=pd.DataFrame(element_rows),
        reactions=pd.DataFrame(reaction_rows),
        loads=pd.DataFrame(loads_rows),
        load_breakdown=breakdown,
        max_uz_mm=max_uz_mm,
        max_abs_n_kn=max_abs_n_kn,
        total_reaction_z_n=total_reaction_z,
        artificial_reactions=pd.DataFrame(artificial_rows),
        scenario=data.snow_pattern,
    )
    if data.spatial_frame:
        result.equilibrium = spatial_equilibrium(model, result)
    result.checks = assessment(data, model, result)
    return result


def spatial_equilibrium(model, result):
    """Global force AND moment balance about origin, including base moments."""
    import numpy as np
    total = np.zeros(6)
    magnitude = np.zeros(6)
    for row in result.loads.itertuples():
        f = np.array([0.0, 0.0, row.Fz_kN])
        v = np.r_[f, np.cross(model.nodes[row.node].coords, f)]
        total += v
        magnitude += np.abs(v)
    for row in result.reactions.itertuples():
        f = np.array([row.Rx_kN, row.Ry_kN, row.Rz_kN])
        moment = np.array([row.Mx_kNm, row.My_kNm, row.Mz_kNm])
        v = np.r_[f, moment + np.cross(model.nodes[row.base_node].coords, f)]
        total += v
        magnitude += np.abs(v)
    tolerance = 1e-6 + 1e-8 * magnitude
    if not np.isfinite(total).all() or (np.abs(total) > tolerance).any():
        raise ValueError(f"Spatial equilibrium failed: {total}")
    return pd.DataFrame({"Компонента": ["Fx, кН", "Fy, кН", "Fz, кН", "Mx, кН·м", "My, кН·м", "Mz, кН·м"],
                         "Невязка": total, "Допуск": tolerance})


def _failed_result(
    data: CanopyInput,
    model: StructuralModel,
    steel: SteelMaterial,
    snow_loads: dict[int, float],
    dead_loads: dict[int, float],
    self_loads: dict[int, float],
    nodal_loads: dict[int, float],
    mu: float,
    s0_kpa: float,
    sd_kpa: float,
    code: int,
) -> FEMResult:
    result = FEMResult(
        ok=False,
        nodes=pd.DataFrame(),
        elements=pd.DataFrame(),
        reactions=pd.DataFrame(),
        loads=pd.DataFrame([{"node": node_id, "Fz_kN": fz / 1000.0} for node_id, fz in sorted(nodal_loads.items())]),
        load_breakdown=_load_breakdown(
            data, model, steel, mu, s0_kpa, sd_kpa,
            snow_loads, dead_loads, self_loads, nodal_loads, 0.0,
        ),
        max_uz_mm=0.0,
        max_abs_n_kn=0.0,
        total_reaction_z_n=0.0,
        message=f"OpenSees: код {code}. Решение не получено; возможен механизм или плохая обусловленность. Искусственная стабилизация не добавлялась.",
        scenario=data.snow_pattern,
    )

    result.checks = assessment(data, model, result)
    return result
