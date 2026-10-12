"""Reproducible partial audit; run: PYTHONPATH=. .venv/bin/python scripts/audit_project.py.

Optional --input accepts an exported app JSON (input object) or CanopyInput dict.
Never promotes partial response checks to project approval.
"""
import argparse
from dataclasses import asdict, replace
from html import escape
import json
from pathlib import Path

import pandas as pd
from src.config import input_from_preset
from src.models import CanopyInput
from src.geometry.naves_one import generate_naves_one
from src.fem.opensees_solver import solve_model
from src.loads.scenarios import PATTERNS
from src.checks.project_audit import span_responses, roof_response, euler_length_thresholds

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path)
parser.add_argument('--output', type=Path, default=Path('outputs/reports/three-checks'))
args = parser.parse_args()
if args.input:
    payload = json.loads(args.input.read_text())
    data = CanopyInput(**payload.get('input', payload))
else:
    data = replace(input_from_preset('naves_one'), fem_subdivisions=4)
if not data.spatial_frame or data.preset_name != 'naves_one':
    raise ValueError('Audit requires spatial naves_one model')
args.output.mkdir(parents=True, exist_ok=True)
(args.output/'input.json').write_text(json.dumps(asdict(data), ensure_ascii=False, indent=2))
model = generate_naves_one(data)
summary, forces, spans, roofs, reactions, euler = [], [], [], [], [], []
for gamma in (data.snow_gamma_f, 1.):
    for pattern, title in PATTERNS.items():
        case = f'{title}; снег γf={gamma:g}'
        result = solve_model(replace(data, snow_pattern=pattern, snow_gamma_f=gamma), model)
        if not result.ok:
            raise RuntimeError(f'{case}: {result.message}')
        worst = result.elements.loc[result.elements.elastic_NM_util.idxmax()]
        summary.append(dict(case=case, min_Uz_mm=result.max_uz_mm, max_N_kN=result.max_abs_n_kn,
                            max_Rz_kN=result.reactions.Rz_kN.max(), axial_util=result.elements.utilization.max(),
                            NM_util=worst.elastic_NM_util, worst_member=worst.label,
                            normal_stress_status='EXCEEDED' if worst.elastic_NM_util > 1 else 'PARTIAL_ONLY'))
        forces.append(result.elements.nlargest(10, 'elastic_NM_util').assign(case=case))
        reactions.append(result.reactions.assign(case=case))
        euler.append(euler_length_thresholds(result).assign(case=case))
        if gamma == 1.:
            spans.append(span_responses(model, result).assign(case=case))
            roofs.append(roof_response(data, model, result).assign(case=case))
        print(case, round(result.max_uz_mm, 3), round(worst.elastic_NM_util, 3), flush=True)
summary = pd.DataFrame(summary)
tables = dict(cases=summary, critical_members=pd.concat(forces), reactions=pd.concat(reactions),
              euler_thresholds=pd.concat(euler), spans=pd.concat(spans), roof_slopes=pd.concat(roofs),
              missing_checks=result.checks)
for name, table in tables.items():
    table.to_csv(args.output/f'{name}.csv', index=False)
roof_summary = tables['roof_slopes'].groupby('case').agg(min_slope_percent=('slope_percent','min'),
                  below_threshold=('below_0_5_percent','sum')).reset_index()
span_summary = tables['spans'].sort_values('max_abs_relative_mm', ascending=False).head(20)
euler_summary = tables['euler_thresholds'].sort_values('Euler_Lcrit_weak_m').head(20)
html = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Три проверки навеса</title>
<style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#17202a}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:8px;border:1px solid #ccd4dc;text-align:left}h2{margin-top:36px}.alert{padding:20px;background:#fff0e8;border-left:5px solid #a30}pre{white-space:pre-wrap}</style>
<h1>Проверка доступной пространственной модели</h1>
<p class="alert"><b>Пригодность к строительству не подтверждена.</b> Это частичная проверка модели, не полный расчёт по СП. Проверен вариант из приложенного input.json; параметры и ограничения приведены ниже.</p>
<h2>1. Прочность и устойчивость</h2><p>Проверены упругие нормальные напряжения от N, My, Mz. Значение выше 1 — превышение в принятой модели. Срез, кручение, местная и общая устойчивость, взаимодействие сжатия и изгиба по СП 16, второй порядок и соединения не проверены. Половинные загружения — исследовательские, не замена нормативным схемам. Коэффициенты постоянных нагрузок здесь не назначены: это существующие нагрузки модели, не полный набор расчётных сочетаний; ветер отсутствует.</p>'''
html += summary.to_html(index=False, float_format=lambda x:f'{x:.3f}')
html += '<h3>Критические элементы</h3>' + tables['critical_members'].sort_values('elastic_NM_util', ascending=False).head(15).to_html(index=False)
html += '<h3>Индикатор устойчивости: предельная расчётная длина по Эйлеру</h3><p>Euler_Lcrit_weak_m — длина, при которой идеальная шарнирная колонна по слабой оси теряет устойчивость при данном сжатии. Это не допустимая длина по СП. Реальная расчётная длина неизвестна; длина конечного элемента не используется вместо неё. Наличие изгиба и несовершенств требует отдельной проверки. Для поясов берётся наибольшее сжатие группы, а не сумма сил.</p>' + euler_summary.to_html(index=False, float_format=lambda x:f'{x:.3f}')
html += '<h2>2. Прогибы и водоотвод</h2><p>Ниже реакции на постоянный вес и полный нормативный снег γf=1, включая исследовательские половинные схемы. Это не автоматический выбор эксплуатационного сочетания по виду требования. Прогиб нижнего пояса измерен относительно прямой между переместившимися опорами каждого пролёта. Консоли и прогиб основания мембраны не проверены. L_over_f — измеренное отношение, не назначенный предел. Предельные прогибы и соответствующие сочетания ещё требуется установить.</p>' + span_summary.to_html(index=False, float_format=lambda x:f'{x:.3f}')
html += '<p>Уклон вычислен по треугольной интерполяции деформированных верхних поясов в исходном направлении стока. Контрольный ориентир 0,5% взят из СП 20 п.15.1.4; требования конкретной мембранной системы проверяются отдельно. Положительный уклон интерполированной поверхности не доказывает отсутствие луж: основание между фермами, водостоки, швы и монтажные отклонения отсутствуют в модели.</p>' + roof_summary.to_html(index=False, float_format=lambda x:f'{x:.3f}')
html += '<h2>3. Соответствие строительству</h2><p>Подтверждена только согласованность заданной геометрии и принятых гипотез. Жёсткая заделка оснований и абсолютно жёсткие сварные узлы приняты условно. Не заданы: подтверждённая сталь; состав/вес и несущая способность основания мембраны; размеры и прочность сварных узлов; расчётные длины и боковые закрепления; Жёсткие основания приняты по согласованной гипотезе, их прочность не входит в эту проверку металлокаркаса. Все одновременные реакции опор сохранены в reactions.csv для отдельного расчёта оснований.</p>' + result.checks.to_html(index=False)
html += '<h2>Исходные параметры</h2><pre>' + escape(json.dumps(asdict(data), ensure_ascii=False, indent=2)) + '</pre><p>Нормативные основания: <a href="https://internet-law.ru/documents/prod/pravila/0/sp_20.html">СП 20, раздел 15, приложение Д</a>; <a href="https://nav.tn.ru/cloud/iblock/9f7/9f78782393359c071745ad3c74d9179f/SP-16.13330.2017-Stalnye-konstruktsii.pdf">СП 16, разделы 7–9</a>.</p></html>'
(args.output/'report.html').write_text(html)
print('ROOF', roof_summary.to_string(index=False))
print('SPANS', span_summary[['group','L_m','max_abs_relative_mm','L_over_f']].head(5).to_string(index=False))
print('EULER', euler_summary[['label','Euler_Lcrit_weak_m']].head(5).to_string(index=False))
