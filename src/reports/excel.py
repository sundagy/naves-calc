from __future__ import annotations

from io import BytesIO

import pandas as pd

from src.fem.opensees_solver import FEMResult


def results_xlsx(result: FEMResult) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.nodes.to_excel(writer, sheet_name="Nodes", index=False)
        result.elements.to_excel(writer, sheet_name="Elements", index=False)
        result.reactions.to_excel(writer, sheet_name="Reactions", index=False)
        result.loads.to_excel(writer, sheet_name="Loads", index=False)
    return output.getvalue()

