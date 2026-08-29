from __future__ import annotations

import pandas as pd


def max_vertical_deflection(nodes: pd.DataFrame) -> dict:
    if nodes.empty:
        return {"node": None, "Uz_mm": 0.0}
    row = nodes.sort_values("Uz_mm").iloc[0]
    return row.to_dict()

