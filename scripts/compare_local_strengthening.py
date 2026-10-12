"""Compare whole-chord changes without modifying the active project preset."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import pandas as pd

from src.config import input_from_preset
from src.geometry.naves_one import generate_naves_one
from src.fem.opensees_solver import solve_model
from src.loads.scenarios import PATTERNS
from src.loads.self_weight import mass_breakdown
from src.checks.project_audit import euler_length_thresholds

out = Path('outputs/reports/local-strengthening')
out.mkdir(parents=True, exist_ok=True)
base = replace(input_from_preset('naves_one'), fem_subdivisions=4)
variants = {
    'baseline': {},
    'b3-top-4mm': {'B3 top chord': '60x60x4'},
    'b3-b4-top-4mm': {'B3 top chord': '60x60x4', 'B4 top chord': '60x60x4'},
}
summaries, all_rows, limits, schedules = [], [], [], []
for name, overrides in variants.items():
    data = replace(base, member_section_overrides=overrides)
    (out/f'{name}-input.json').write_text(json.dumps(asdict(data), ensure_ascii=False, indent=2))
    model = generate_naves_one(data)
    for label in overrides:
        elements = [e for e in model.elements if e.label == label]
        ids = {n for e in elements for n in (e.i,e.j)}
        schedules.append(dict(variant=name,label=label,section=overrides[label],
                             y_start=min(model.nodes[n].y for n in ids),
                             y_end=max(model.nodes[n].y for n in ids)))
    for pattern in PATTERNS:
        result = solve_model(replace(data, snow_pattern=pattern), model)
        if not result.ok:
            raise RuntimeError(result.message)
        worst = result.elements.loc[result.elements.elastic_NM_util.idxmax()]
        summaries.append(dict(variant=name,case=pattern,mass_kg=mass_breakdown(model)['total'],
                              Uz_mm=result.max_uz_mm,NM_max=worst.elastic_NM_util,worst=worst.label))
        elements = result.elements.assign(variant=name,case=pattern).copy()
        elements['bending_util_at_critical_end'] = elements.elastic_NM_util-elements.utilization
        all_rows.append(elements)
        limits.append(euler_length_thresholds(result).assign(variant=name,case=pattern))
        print(name,pattern,round(worst.elastic_NM_util,4),worst.label,flush=True)
summary = pd.DataFrame(summaries)
all_elements = pd.concat(all_rows,ignore_index=True)
summary.to_csv(out/'comparison.csv',index=False)
all_elements.to_csv(out/'all-elements.csv',index=False)
pd.concat(limits).to_csv(out/'euler-length-thresholds.csv',index=False)
pd.DataFrame(schedules).to_csv(out/'chord-changes.csv',index=False)
# Physical label envelopes; chord chains are not treated as FE-length columns.
envelope = all_elements.sort_values('elastic_NM_util',ascending=False).drop_duplicates(['variant','group','label'])
envelope.to_csv(out/'member-envelope.csv',index=False)
print(summary.groupby('variant').agg(NM_max=('NM_max','max'),Uz_mm=('Uz_mm','min'),mass_kg=('mass_kg','max')).to_string())

names = {'baseline':'Текущий вариант','b3-top-4mm':'Верхний пояс B3: 60×60×4',
         'b3-b4-top-4mm':'Верхние пояса B3 и B4: 60×60×4'}
compact = summary.groupby('variant').agg(NM_max=('NM_max','max'),Uz_mm=('Uz_mm','min'),mass_kg=('mass_kg','max'))
compact['added_mass_kg'] = compact.mass_kg-compact.loc['baseline','mass_kg']
compact = compact.rename(index=names, columns={'NM_max':'Макс. N–M/Ry','Uz_mm':'Min Uz, мм',
                                             'mass_kg':'Масса, кг','added_mass_kg':'Добавлено, кг'})
chosen = envelope[envelope.variant=='b3-top-4mm'][['label','section','N_kN','utilization',
                       'bending_util_at_critical_end','elastic_NM_util']].head(12)
chosen = chosen.rename(columns={'label':'Элемент','section':'Труба','N_kN':'N, кН',
                       'utilization':'Вклад N','bending_util_at_critical_end':'Вклад изгиба',
                       'elastic_NM_util':'Сумма N–M/Ry'})
fmt = lambda x:f'{x:.3f}'
html = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Адресное усиление навеса</title>
<style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd4dc;padding:9px;text-align:left}aside{padding:18px;background:#fff1dd}h2{margin-top:35px}</style>
<h1>Сравнение адресного усиления</h1>
<p>Исходный вариант: нижние фермы 700 мм, пояса и нижняя решётка 60×60×3; верхние фермы 400→170 мм, решётка 40×40×3. Уклон 1,072°. Стойки 120×120×3. Полный условный вес кровли 0,15 кН/м². Жёсткие узлы и основания — гипотезы.</p>
<aside>Сравнение упругих нормальных напряжений, не допуск к строительству. Ветер, нормативная устойчивость, местная прочность труб в узлах, швы и основания не проверены. Действующий пресет не изменён; варианты сохранены отдельными JSON.</aside>
<h2>Что найдено</h2><p>Во всех пяти исследованных снеговых схемах порог поиска 0,8 превышен только у верхнего пояса B3: 0,841 = 0,611 от сжатия + 0,231 от изгиба (округлено). Превышений 1 нет. Порог 0,8 — ориентир поиска, не норматив.</p>
<h2>Сравнение вариантов</h2>'''
html += compact.to_html(float_format=fmt)
html += '''<h2>Предпочтительный вариант для дальнейшей проверки</h2>
<p>Заменить весь верхний пояс B3 длиной 8,725 м с 60×60×3 на 60×60×4. Это замена трубы при изготовлении, а не наваривание накладок и не указание на выполнение безстыковой трубы такой длины: стыки и сварные узлы требуют деталировки. Высоты, наружные размеры и уклон сохранены. Добавляется около 14,5 кг стали.</p>
<p>Усиление B4 дополнительно добавляет около 13,1 кг и уменьшает общий максимум лишь с 0,741 до 0,722. По данному критерию ограничиваем предложение поясом B3; решение по устойчивости может потребовать других изменений.</p>
<h2>Перераспределение в варианте B3</h2>'''
html += chosen.to_html(index=False,float_format=fmt)
html += '''<p>Сжатие усиленного пояса выросло с 102,3 до 109,5 кН; его N–M/Ry снизилось до 0,694. Общий максимум 0,741 перешёл в нижний пояс B3. Это показывает, почему усиление требует пересчёта всего каркаса.</p>
<h2>Что известно об устойчивости</h2>
<p>Индикатор Эйлера для верхнего пояса B3 даёт предельную расчётную длину около 2,68 м до замены и 2,91 м после неё. Это длина при равенстве идеальной эйлеровой силы текущему сжатию, не допустимая длина по СП и не подтверждённое расстояние между связями. Фактическая расчётная длина из плоскости фермы не установлена; пересечение с другой фермой само по себе не доказывает достаточное раскрепление. Все сжатые группы включены в euler-length-thresholds.csv.</p>
<h2>Нагрузки и ограничения</h2><p>Собственный вес обновлён для каждого сечения. Выполнены равномерное и четыре половинных снеговых загружения при γf=1,4; половинные схемы исследовательские. Расчёт линейный, постоянные нагрузки без отдельного подбора коэффициентов сочетаний. Это не полный набор нормативных сочетаний. Ry=245 МПа условно. Прогибы в таблице абсолютные, не проверка допустимости.</p>
<p>Узлы требуют отдельной проверки стенок пояса, примыкающих элементов и сварных швов: <a href="https://internet-law.ru/documents/prod/pravila/0/sp_16.html">СП 16, п.15.2.5</a>.</p>
<p><a href="b3-top-4mm-input.json">Параметры варианта B3</a> · <a href="comparison.csv">Все расчётные случаи</a> · <a href="member-envelope.csv">Максимумы по группам элементов</a></p></html>'''
(out/'report.html').write_text(html)
