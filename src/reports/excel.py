from __future__ import annotations

from io import BytesIO
from dataclasses import asdict

import pandas as pd

from src.fem.opensees_solver import FEMResult


def results_xlsx(result: FEMResult) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(
            [(name, value) for name, value in asdict(result.load_breakdown).items()],
            columns=["Показатель", "Значение"],
        ).to_excel(writer, sheet_name="Summary", index=False)
        result.equilibrium.to_excel(writer, sheet_name="Spatial equilibrium", index=False)
        result.checks.to_excel(writer, sheet_name="Check coverage", index=False)
        result.scenarios.to_excel(writer, sheet_name="Snow scenarios", index=False)
        result.nodes.to_excel(writer, sheet_name="Nodes", index=False)
        result.elements.to_excel(writer, sheet_name="Elements", index=False)
        result.reactions.to_excel(writer, sheet_name="Reactions", index=False)
        result.artificial_reactions.to_excel(writer, sheet_name="Artificial restraints", index=False)
        result.loads.to_excel(writer, sheet_name="Loads", index=False)
    return output.getvalue()
