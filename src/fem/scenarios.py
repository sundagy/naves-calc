from dataclasses import replace

import pandas as pd

from src.fem.opensees_solver import solve_model
from src.loads.scenarios import PATTERNS


def solve_snow_suite(data, model):
    """Keep each case separate; maxima are not a concurrent force combination."""
    rows = []
    for pattern in PATTERNS:
        result = solve_model(replace(data, snow_pattern=pattern), model)
        rows.append({
            "Сценарий": PATTERNS[pattern], "Решение": result.ok,
            "Снег, кН": result.load_breakdown.snow_load_total_kN,
            "Min Uz, мм": result.max_uz_mm if result.ok else None,
            "Max |N|, кН": result.max_abs_n_kn if result.ok else None,
            "Max N–M": result.elements.elastic_NM_util.max() if result.ok else None,
            "Сообщение": result.message,
        })
    # Reference load level only; not an automated deflection acceptance check.
    normative = solve_model(replace(data, snow_pattern="UNIFORM", snow_gamma_f=1.0), model)
    rows.append({"Сценарий": "Нормативный равномерный снег (γf=1)", "Решение": normative.ok,
                 "Снег, кН": normative.load_breakdown.snow_load_total_kN,
                 "Min Uz, мм": normative.max_uz_mm if normative.ok else None,
                 "Max |N|, кН": normative.max_abs_n_kn if normative.ok else None,
                 "Max N–M": normative.elements.elastic_NM_util.max() if normative.ok else None,
                 "Сообщение": normative.message})
    return pd.DataFrame(rows)
