from __future__ import annotations

from collections import defaultdict
from math import dist

from src.materials.steel import SteelMaterial
from src.models import Element, StructuralModel
from src.sections.database import section_database


G_ACCEL = 9.80665


def element_length(model: StructuralModel, element: Element) -> float:
    ni = model.nodes[element.i]
    nj = model.nodes[element.j]
    return dist(ni.coords, nj.coords)


def element_mass(model: StructuralModel, element: Element, steel: SteelMaterial | None = None) -> float:
    material = steel or SteelMaterial()
    section = section_database()[element.section]
    return section.area * element_length(model, element) * material.rho


def mass_breakdown(model: StructuralModel, steel: SteelMaterial | None = None) -> dict[str, float]:
    result = {"lower": 0.0, "upper": 0.0, "columns": 0.0, "crossbeams": 0.0, "total": 0.0}
    for element in model.elements:
        mass = element_mass(model, element, steel)
        if element.role.startswith("lower"):
            result["lower"] += mass
        elif element.role.startswith("upper"):
            result["upper"] += mass
        elif element.role == "column":
            result["columns"] += mass
        elif element.role == "crossbeam":
            result["crossbeams"] += mass
        result["total"] += mass
    result["upper"] += model.additional_dead_mass_kg
    result["total"] += model.additional_dead_mass_kg
    return result


def self_weight_nodal_loads(model: StructuralModel, steel: SteelMaterial | None = None) -> dict[int, float]:
    loads = defaultdict(float)
    for element in model.elements:
        weight = element_mass(model, element, steel) * G_ACCEL
        loads[element.i] -= weight / 2.0
        loads[element.j] -= weight / 2.0
    if model.additional_dead_mass_kg:
        if not model.roof_tributary_areas:
            raise ValueError("Supplementary mass requires roof tributary areas")
        area = sum(model.roof_tributary_areas.values())
        for node_id, node_area in model.roof_tributary_areas.items():
            loads[node_id] -= model.additional_dead_mass_kg * G_ACCEL * node_area / area
    return dict(loads)
