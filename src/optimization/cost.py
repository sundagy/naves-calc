from __future__ import annotations

from src.loads.self_weight import mass_breakdown
from src.models import CanopyInput, StructuralModel


def cost_breakdown(data: CanopyInput, model: StructuralModel) -> dict[str, float]:
    masses = mass_breakdown(model)
    steel_cost = masses["total"] * data.steel_price_per_kg
    fabrication = (data.lower_count + data.upper_count) * data.fabrication_per_truss
    columns = len(model.columns) * data.column_cost
    foundations = len(model.columns) * data.foundation_cost
    connections = len(model.elements) * data.connection_cost
    return {
        "steel_cost": steel_cost,
        "fabrication_cost": fabrication,
        "column_cost": columns,
        "foundation_cost": foundations,
        "connection_cost": connections,
        "total_cost": steel_cost + fabrication + columns + foundations + connections,
    }

