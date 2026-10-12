from __future__ import annotations

from collections import defaultdict

from src.geometry.canopy import CanopyPlan, upper_truss_positions
from src.models import CanopyInput, StructuralModel


def integrate_depth(plan: CanopyPlan, x1: float, x2: float) -> float:
    d1 = plan.depth(x1)
    d2 = plan.depth(x2)
    return (d1 + d2) * (x2 - x1) / 2.0


def tributary_roof_loads(
    data: CanopyInput,
    model: StructuralModel,
    pressure_kpa: float,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> dict[int, float]:
    if model.roof_tributary_areas is not None:
        if t_min != 0.0 or t_max != 1.0:
            raise ValueError("Partial snow zones are not defined for the NavesOne outline")
        return {node_id: -pressure_kpa * 1000.0 * area for node_id, area in model.roof_tributary_areas.items()}
    plan = CanopyPlan(width=data.width, left_depth=data.left_depth, right_depth=data.right_depth)
    upper_ts = upper_truss_positions(data)
    loads = defaultdict(float)
    pressure_pa = pressure_kpa * 1000.0

    for t_index, station_t in enumerate(upper_ts):
        upper_name = f"U{t_index + 1}"
        upper_node_ids = model.upper_trusses[upper_name]
        upper_xs = [model.nodes[node_id].x for node_id in upper_node_ids]
        low = 0.0 if t_index == 0 else (upper_ts[t_index - 1] + station_t) / 2.0
        high = 1.0 if t_index == len(upper_ts) - 1 else (station_t + upper_ts[t_index + 1]) / 2.0
        low = max(low, t_min)
        high = min(high, t_max)
        if high <= low:
            continue
        for x_index, node_id in enumerate(upper_node_ids):
            x_left = 0.0 if x_index == 0 else (upper_xs[x_index - 1] + upper_xs[x_index]) / 2.0
            x_right = data.width if x_index == len(upper_xs) - 1 else (upper_xs[x_index] + upper_xs[x_index + 1]) / 2.0
            area = integrate_depth(plan, x_left, x_right) * (high - low)
            force = pressure_pa * area
            loads[node_id] -= force
    return dict(loads)
