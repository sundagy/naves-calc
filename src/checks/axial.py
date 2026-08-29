from __future__ import annotations

import pandas as pd


def top_utilized(elements: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    if elements.empty:
        return elements
    return elements.sort_values("utilization", ascending=False).head(count)

