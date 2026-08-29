from __future__ import annotations

from src.geometry.canopy import CanopyPlan, lower_truss_positions
from src.models import CanopyInput, SupportMode


EXISTING_10_COLUMNS: dict[str, tuple[float, float]] = {
    "C1": (0.0, 0.0),
    "C2": (4.0, 0.0),
    "C3": (8.0, 0.0),
    "C4": (12.0, 0.0),
    "C5": (0.0, 10.0),
    "C6": (4.0, 9.0),
    "C7": (8.0, 8.0),
    "C8": (12.0, 7.0),
    "C9": (0.0, 3.5),
    "C10": (12.0, 3.5),
}

ORIGINAL_SUPPORTS_BY_TRUSS: dict[str, list[str]] = {
    "F1": ["C1", "C9", "C5"],
    "F2": ["C2", "C6"],
    "F3": ["C3", "C7"],
    "F4": ["C4", "C10", "C8"],
}


def build_column_points(data: CanopyInput, plan: CanopyPlan) -> dict[str, tuple[float, float]]:
    if data.support_mode == SupportMode.EXISTING_10:
        return dict(EXISTING_10_COLUMNS)
    if data.support_mode == SupportMode.LOWER_TRUSS_ENDS:
        points: dict[str, tuple[float, float]] = {}
        for index, x in enumerate(lower_truss_positions(data), start=1):
            points[f"F{index}_front"] = (x, 0.0)
            points[f"F{index}_rear"] = (x, plan.depth(x))
        points["C9"] = EXISTING_10_COLUMNS["C9"]
        points["C10"] = EXISTING_10_COLUMNS["C10"]
        return points
    raise NotImplementedError("CUSTOM support mode is reserved for UI/API extension.")


def validate_columns(data: CanopyInput, plan: CanopyPlan, columns: dict[str, tuple[float, float]]) -> None:
    if not data.allow_interior_columns:
        for name, (x, y) in columns.items():
            if plan.contains_strictly(x, y):
                raise ValueError("Interior columns are disabled for this project.")
            if not plan.on_boundary(x, y):
                raise ValueError(f"Column {name} is not on the trapezoid perimeter.")

    if data.preset_name == "original" and data.support_mode == SupportMode.EXISTING_10:
        if len(columns) != 10:
            raise ValueError("Preset original must contain exactly 10 columns.")
        for name, coords in EXISTING_10_COLUMNS.items():
            if columns.get(name) != coords:
                raise ValueError(f"Preset original column {name} must be {coords}.")
