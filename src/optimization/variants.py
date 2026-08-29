from __future__ import annotations

from dataclasses import asdict

from src.fem.opensees_solver import FEMResult
from src.models import CanopyInput, StructuralModel
from src.optimization.cost import cost_breakdown


def variant_summary(name: str, data: CanopyInput, model: StructuralModel, result: FEMResult) -> dict:
    cost = cost_breakdown(data, model)
    return {
        "name": name,
        "lower_count": data.lower_count,
        "lower_height_mm": data.lower_height_mm,
        "lower_chord_section": data.lower_chord_section,
        "upper_count": data.upper_count,
        "upper_height_mm": data.upper_height_mm,
        "upper_chord_section": data.upper_chord_section,
        "columns": len(model.columns),
        "mass_kg": float(result.elements["mass_kg"].sum()) if result.ok and not result.elements.empty else 0.0,
        "max_uz_mm": result.max_uz_mm,
        "max_utilization": float(result.elements["utilization"].max()) if result.ok and not result.elements.empty else 0.0,
        "cost_rub": cost["total_cost"],
        "input": asdict(data),
    }

