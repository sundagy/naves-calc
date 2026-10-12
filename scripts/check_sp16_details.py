"""Reproducible, explicitly conditional checks of saved snow analysis.

Not a full SP16 implementation: no inferred effective lengths, welds or deck.
Run with .venv/bin/python -m scripts.check_sp16_details.
"""
from pathlib import Path
import json
from math import sqrt, hypot, dist
from html import escape
import numpy as np
import pandas as pd
from dataclasses import replace
from src.sections.rhs import parse_rhs
from src.models import CanopyInput
from src.geometry.naves_one import generate_naves_one

OUT = Path('outputs/reports/snow-completion')
SOURCE = 'https://nav.tn.ru/cloud/iblock/9f7/9f78782393359c071745ad3c74d9179f/SP-16.13330.2017-Stalnye-konstruktsii.pdf'
# SP16 Table D.3 subset, values /1000. No extrapolation beyond this subset.
LAM = np.array([.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
ECC = np.array([.1, .25, .5])
PHI = np.array([[967,922,850],[925,854,778],[875,804,716],
                [813,742,653],[742,672,587],[667,597,520],
                [587,522,455],[505,447,394]]) / 1000


def phi_e(lam, m):
    if not .5 <= lam <= 4 or not 0 <= m <= .5:
        raise ValueError('Outside verified table subset')
    # For m<0.1 take the 0.1 column: conservative, NOT extrapolation to zero.
    return float(np.interp(max(m,.1), ECC,
                          [np.interp(lam,LAM,PHI[:,i]) for i in range(3)]))


def critical_b3_check(row, length):
    s = parse_rhs(row.section)
    N = -row.N_kN * 1000
    assert N > 0 and s.width == s.height
    Ry, E = 245e6, 200e9
    lam = length / s.rx * sqrt(Ry/E)
    # SP axes: Mx corresponds to OpenSees Mz (bending in depth direction).
    mx = row.Mz_max_kNm*1000*s.area/(N*s.wx)
    my = row.My_max_kNm*1000*s.area/(N*s.wy)
    def phi(m):
        # D.2 type 4, lambda<=5 and m<=5.
        eta = (1.35-.05*m)-.01*(5-m)*lam
        return phi_e(lam,eta*m)
    delta = 1 if lam <= 1 else 1-.1*N*lam**2/(s.area*Ry)
    assert delta > 0
    # c_x=c_y=1 conservatively omits plastic reserve (E.1 maxima >1).
    # gamma_c=1 is an explicit assumption, not an assigned project coefficient.
    u120 = N/(phi(my)*s.area*Ry)+row.Mz_max_kNm*1000/(delta*s.wx*Ry)
    u121 = N/(phi(mx)*s.area*Ry)+row.My_max_kNm*1000/(delta*s.wy*Ry)
    # Local check for this section only, 0<mx<1: table22 note interpolation;
    # table23 type2; table9 box, lambda>1. No use of unread formula25.
    assert 1 < lam <= 4 and 0 < mx < 1
    central = min(1+.2*lam,1.6)
    wall_at_m1 = 1.3+.15*lam**2 if lam < 2 else min(1.2+.35*lam,3.1)
    wall_limit = central+(wall_at_m1-central)*mx
    flange_limit = central-.01*(5.3+1.3*lam)*mx
    plate = (s.width-2*s.thickness)/s.thickness*sqrt(Ry/E)
    return dict(L_eff_m=length,lambda_bar=lam,mx=mx,my=my,
                condition120=u120,condition121=u121,
                governing=max(u120,u121),plate_lambda=plate,
                flange_limit=flange_limit,wall_limit=wall_limit)


def main():
    d = pd.read_csv(OUT/'UNIFORM-q0.15-mesh4-elements.csv')
    critical = d.loc[d.elastic_NM_util.idxmax()]
    assert critical.label == 'B3 top chord' and critical.section == '80x80x3'
    b3 = pd.DataFrame([critical_b3_check(critical,l) for l in (1.,1.5,2.,2.5,3.)])
    b3.to_csv(OUT/'B3-conditional-SP16.csv',index=False)
    forces = pd.read_csv(OUT/'all-snow-joint-demands.csv')
    web = forces[forces.label.eq('B3 vertical 1')].copy()
    web['Mb_kNm'] = np.hypot(web.My_kNm,web.Mz_kNm)
    web = web.loc[[web.Mb_kNm.idxmax()]]
    web.to_csv(OUT/'B3-vertical-joint-demand.csv',index=False)
    inputs=CanopyInput(**json.loads(Path('outputs/reports/frame-600-fixed/input.json').read_text()))
    model=generate_naves_one(replace(inputs,fem_subdivisions=1))
    # Trace high V/T to geometry, not an automatic section failure assertion.
    short=[]
    for e in model.elements:
        length=dist(model.nodes[e.i].coords,model.nodes[e.j].coords)
        if length < .001:
            short.append(dict(element=e.id,label=e.label,i=e.i,j=e.j,length_mm=1000*length))
    shorts=pd.DataFrame(short)
    shorts.to_csv(OUT/'submillimetre-members.csv',index=False)
    deck=[]
    for L in (1.,1.5,2.):
        # 1m wide simple strip, model loads. Not a final deck design.
        q_uls=1.9*1.4+.15
        q_sls=1.9+.15
        M=q_uls*L*L/8
        V=q_uls*L/2
        I=5*q_sls*(L*1000)**4/(384*200000*(L*1000/200))
        deck.append(dict(span_m=L,q_model_kPa=q_uls,M_kNm_per_m=M,V_kN_per_m=V,
                         W_elastic_required_cm3_per_m=M*1e6/245/1000,
                         I_for_assumed_L200_cm4_per_m=I/10000))
    deck=pd.DataFrame(deck);deck.to_csv(OUT/'deck-design-demands.csv',index=False)
    tables=lambda x:x.to_html(index=False,float_format=lambda v:f'{v:.4f}')
    html='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Уточнение проверок по снегу</title>
<style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#172537}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{padding:8px;border:1px solid #bbb}.note{background:#fff1db;padding:18px}h2{margin-top:32px}</style>
<h1>Сжатие с изгибом, стенки, узлы и основание мембраны</h1>
<p class="note"><b>Окончательное соответствие проекта не установлено.</b> Ниже — новые условные численные проверки и конкретные препятствия для завершения. «Условно проходит» относится только к указанному сечению, нагрузке и расчётной длине, а не ко всему навесу.</p>
<p>Исходный вариант: B высотой 600 мм, пояса 80×80×3, решётка 60×60×3; F — 60×60×3 / 40×40×3, стойки 120×120×3. Основания жёсткие по принятой гипотезе. Надземные узлы в модели абсолютно жёсткие. Ry=245 МПа, E=200 ГПа — условные исходные значения. Здесь нагрузки и сечения не изменены.</p>
<h2>1. Проверка наиболее напряжённого сечения B3</h2>
<p>Равномерный расчётный снег: N=116,623 кН сжатия, Mx=1,02625 кН·м, My=0,00584 кН·м. Моменты — максимумы концов одного КЭ, сочетание консервативное. Площадь 924 мм², момент сопротивления 22861,3 мм³. Формулы СП16 9.2.10 (120), (121); коэффициенты таблиц Д.2 тип 4 и Д.3. Приняты γc=1 и консервативно cx=cy=1. При m&lt;0,1 берётся столбец m=0,1, без экстраполяции; в остальных случаях билинейная интерполяция.</p>
<p><b>Расчётные длины в таблице — сценарии, а не установленные длины реальных элементов.</b> В обеих плоскостях принята одинаковая длина. Нельзя подставлять длину конечного элемента. Проверка одного наиболее напряжённого сечения не выявляет автоматически самый опасный по устойчивости стержень.</p>'''
    html+=tables(b3)
    html+='''<p>condition120/121 ≤1 — выполнение соответствующего условия при выбранной длине. Значение выше 1 при выбранном консервативном cx=cy=1 означает невыполнение этой проверки, а не доказанное разрушение. Для окончательной проверки всех стержней требуется назначить расчётные длины в обеих плоскостях по фактическому раскреплению (раздел 10 СП16), после чего проверить все сжатые элементы и сочетания.</p>
<h2>2. Местная устойчивость стенок</h2>
<p>В той же таблице plate_lambda = 0,8633 — условная гибкость плоской части трубы 80×80×3. Ширина принята 80−2×3=74 мм, без уменьшения на закругления. flange_limit — таблица 23, тип 2; wall_limit — таблица 22, интерполяция для 0&lt;m&lt;1 с таблицей 9. Для показанных сценариев у сечения B3 обе границы выше фактической гибкости. Это не завершает проверку всех труб, двухосного распределения напряжений и местной работы стенок в узлах.</p>
<h2>3. Соединения: обнаружены существенные моменты</h2>
<p>Один одновременный набор усилий на конце вертикали B3 (кН, кН·м):</p>'''+tables(web)
    html+='''<p>Шов нужно рассчитывать на этот момент вместе с силой, а не только на сжатие 33 кН. По СП16 п.15.2.5 в непосредственных примыканиях труб отдельно проверяются стенка пояса, примыкающая решётка и швы. Катет, длина, контур шва и устройство опирания не заданы, поэтому коэффициент использования соединения не определён. Усилия на концах стержней не всегда равны усилиям в реальном сварном шве.</p>
<p><b>Особенность модели у опоры B1:</b> между узлами 26 и 27 есть участок длиной около 0,52 мм. На нём получены V≈43,86 кН и T≈1,12 кН·м; предварительная балочная оценка N–M–V–T даёт около 0,98 от Ry. Но длина участка намного меньше размера трубы 80 мм: это зона ввода опорной реакции, где балочная формула не подтверждает прочность реального узла. Число 0,98 не следует считать достоверным коэффициентом узла. Нужны его геометрия и местный расчёт; удалять этот участок ради улучшения показателя нельзя.</p>'''+tables(shorts)
    html+='''<h2>4. Что должен выдержать настил под мембраной</h2>
<p>Предварительные требования для полосы настила шириной 1 м, разрезное опирание, равномерная нагрузка. Использованы прежние нагрузки модели: снег 2,66 кПа расчётный / 1,90 кПа нормативный, условный вес кровли 0,15 кПа. Коэффициенты постоянной нагрузки и реальный состав кровли ещё не уточнены; это не окончательная нормативная комбинация.</p>'''+tables(deck)
    html+='''<p>W рассчитан для условной стали Ry=245 МПа; I — для E=200 ГПа и принятого для примера ограничения L/200 (не назначенная норма для выбранной кровельной системы). Эти значения нельзя переносить на древесную плиту или на профнастил без проверки его эффективного сечения, местной устойчивости, опорного смятия, креплений и таблиц изготовителя. Мембрана сама не заменяет несущий настил.</p>
<h2>Статус</h2><ul><li>Сжатие с изгибом: выполнена условная проверка критического по N–M сечения B3; полный набор стержней по СП не закрыт.</li><li>Местная устойчивость: показанное сечение B3 удовлетворяет приведённым проверкам для выбранных сценариев; все элементы и узлы не закрыты.</li><li>Соединения: усилия установлены; несущая способность не определена без конструкции узлов.</li><li>Настил: определены предварительные требования; фактическое изделие неизвестно.</li></ul>
<p>Для двух последних пунктов нужны марка/толщина/пролёт настила и схема швов с размерами. По этим исходным данным нельзя выдавать разрешение на изготовление: результаты следует включить в проверку конструктора перед строительством.</p>'''
    html+=f'<p>Нормативная основа: <a href="{escape(SOURCE)}">СП16 с изменениями 1–6</a>, разделы 9, 10, 14 и п.15.2.5.</p></html>'
    (OUT/'details.html').write_text(html)
    print(b3.to_string(index=False));print(deck.to_string(index=False));print(OUT/'details.html')

if __name__ == '__main__':
    main()
