from __future__ import annotations

from dataclasses import asdict
from html import escape

from src.fem.opensees_solver import FEMResult
from src.fem.assumptions import model_limitations
from src.models import CanopyInput


def html_report(data: CanopyInput, result: FEMResult) -> str:
    warning = (
        "Расчёт предназначен для предварительного анализа и сравнения конструктивных вариантов. "
        "Результаты МКЭ и автоматических проверок не заменяют рабочий проект и проверку инженером-конструктором."
    )
    rows = "".join(f"<li>{escape(str(key))}: {escape(str(value))}</li>" for key, value in asdict(data).items())
    summary = (
        f"<p>Max Uz={result.max_uz_mm:.2f} мм, max |N|={result.max_abs_n_kn:.2f} кН, "
        f"sum Rz={result.total_reaction_z_n / 1000:.2f} кН.</p>"
        if result.ok else "<p>Перемещения, усилия и реакции не определены: решение не получено.</p>"
    )
    return f"""
<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><title>Отчёт naves-calc</title></head>
<body>
<h1>Предварительная расчётная проверка</h1>
<p><strong>{warning}</strong></p>
<h2>Надёжность не подтверждена</h2>
<p>Решение МКЭ: {result.ok}. {escape(result.message)}</p>
{result.checks.to_html(index=False)}
<h2>Снеговые сценарии</h2>{result.scenarios.to_html(index=False)}
<h2>Элементы: осевые силы, изгиб, индикатор Эйлера</h2>{result.elements.to_html(index=False)}
<h2>Реакции оснований, включая моменты</h2>{result.reactions.to_html(index=False)}
<h2>Ограничения расчётной схемы</h2><p>{model_limitations(data)}</p>
<h2>Исходные параметры</h2><ul>{rows}</ul>
<h2>Нагрузки</h2>
<p>Sg={result.load_breakdown.sg_kpa:.3f} кН/м²,
ce={result.load_breakdown.ce:.3f},
ct={result.load_breakdown.ct:.3f},
mu={result.load_breakdown.mu:.3f},
gamma_f={result.load_breakdown.gamma_f:.3f},
S0={result.load_breakdown.s0_kpa:.3f} кН/м²,
Sd={result.load_breakdown.sd_kpa:.3f} кН/м².</p>
<h2>Итоги</h2>
{summary}
<h2>Реакции искусственных закреплений вне оснований</h2>
{result.artificial_reactions.to_html(index=False)}
</body></html>
"""
