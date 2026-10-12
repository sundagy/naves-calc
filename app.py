from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.checks.axial import top_utilized
from src.config import input_from_preset, load_presets, load_sections
from src.fem.opensees_solver import VERTICAL_BALANCE_TOLERANCE_KN, solve_model
from src.geometry.canopy import CanopyPlan, roof_plane_from_input
from src.geometry.columns import ORIGINAL_SUPPORTS_BY_TRUSS
from src.geometry.naves_one import generate_naves_one, roof_area as naves_one_roof_area, upper_truss_dimensions
from src.geometry.truss import generate_model
from src.loads.self_weight import G_ACCEL, mass_breakdown
from src.loads.snow_sp20 import kaluga_source, snow_design_pressure_kpa
from src.models import BaseFixity, CanopyInput, SupportMode, WebType
from src.optimization.cost import cost_breakdown
from src.optimization.variants import variant_summary
from src.reports.excel import results_xlsx
from src.reports.html import html_report
from src.visualization.model3d import model_figure


st.set_page_config(page_title="naves-calc", layout="wide")


def main() -> None:
    st.title("Параметрический МКЭ-стенд автомобильного навеса")
    st.caption(
        "Предварительная расчётная проверка. Результаты МКЭ и автоматических проверок "
        "не заменяют рабочий проект и проверку инженером-конструктором."
    )
    presets = {"naves_one": load_presets()["naves_one"]}
    sections = load_sections()
    if "variants" not in st.session_state:
        st.session_state["variants"] = []

    data = sidebar_inputs(presets, sections)
    plan = CanopyPlan(width=data.width, left_depth=data.left_depth, right_depth=data.right_depth)
    roof = roof_plane_from_input(data)

    st.sidebar.info(f"{kaluga_source()}. Значение Sg редактируемое.")
    st.sidebar.metric("Площадь кровли", f"{naves_one_roof_area() if data.preset_name == 'naves_one' else plan.area:.1f} м²")
    st.sidebar.metric("Высота заднего правого угла", f"{roof.rear_right_height:.2f} м")
    st.sidebar.metric("Макс. угол кровли", f"{roof.max_angle_deg:.2f}°")

    try:
        model = generate_naves_one(data) if data.preset_name == "naves_one" else generate_model(data)
    except Exception as exc:
        st.error(str(exc))
        return

    mode = st.selectbox(
        "Режим визуализации",
        ["GEOMETRY", "REALISTIC", "DEFORMED", "AXIAL_FORCE", "DISPLACEMENT", "UTILIZATION", "REACTIONS", "SNOW_LOAD"],
        index=0,
        format_func=lambda value: "Реалистичный" if value == "REALISTIC" else value,
    )
    scale = st.select_slider("Масштаб деформаций", options=[1, 5, 10, 20, 50, 100], value=20)
    calculate = st.button("Рассчитать", type="primary")

    result = None
    if calculate:
        with st.spinner("OpenSeesPy: линейный статический расчёт..."):
            result = solve_model(data, model)
        st.session_state["last_result"] = result
        st.session_state["last_input"] = data
        st.session_state["last_model"] = model
    else:
        last_input = st.session_state.get("last_input")
        if last_input is not None and asdict(last_input) == asdict(data):
            result = st.session_state.get("last_result")
        elif last_input is not None and mode != "SNOW_LOAD":
            st.info("Параметры изменены. Нажмите «Рассчитать», чтобы обновить нагрузки и реакции.")

    fig = model_figure(data, model, result, mode=mode, deformation_scale=scale)
    camera_name = st.radio("Камера", ["Изометрия", "План", "Спереди", "Сзади", "Слева", "Справа"], horizontal=True)
    projection = st.radio(
        "Проекция", ["Перспективная", "Ортогональная"], horizontal=True,
        help="Ортогональная проекция показывает модель без перспективного уменьшения удалённых элементов.",
    )
    apply_camera(fig, camera_name, "orthographic" if projection == "Ортогональная" else "perspective")
    st.html("""
        <style>
        [data-testid="stLayoutWrapper"]:has(> .st-key-model_viewport) {
            flex: 0 0 auto;
            resize: vertical;
            overflow: hidden;
            min-height: 250px;
        }
        .st-key-model_viewport {
            padding: 0;
            border: 0;
        }
        </style>
    """)
    with st.container(key="model_viewport", height=500, border=True):
        st.plotly_chart(fig, width="stretch", height="stretch")
    st.caption("Потяните за нижний правый угол 3D-вида, чтобы изменить его высоту.")
    if mode == "SNOW_LOAD":
        st.caption(
            "Расчётная снеговая нагрузка обновляется без запуска МКЭ. "
            "Длина и цвет стрелок показывают силу в узле; масштаб длины нормирован "
            "по максимальной силе текущей схемы. Наведите на узел или наконечник, "
            "чтобы увидеть силу и грузовую площадь."
        )

    geometry_summary(model, data)
    if result is not None:
        show_results(data, model, result)

    comparison_panel()


def sidebar_inputs(presets: dict[str, dict], sections: list[str]) -> CanopyInput:
    st.sidebar.header("Параметры")
    preset_name = st.sidebar.selectbox("Preset", list(presets), index=0)
    data = input_from_preset(preset_name)
    if preset_name == "naves_one":
        return naves_one_inputs(data, sections)

    lower_count = st.sidebar.slider("Нижние фермы", 4, 8, int(data.lower_count))
    upper_count = st.sidebar.slider("Верхние фермы", 6, 12, int(data.upper_count))
    lower_height = st.sidebar.number_input("Высота нижних ферм, мм", 150.0, 1200.0, float(data.lower_height_mm), 50.0)
    upper_height = st.sidebar.number_input("Высота верхних ферм, мм", 150.0, 1200.0, float(data.upper_height_mm), 50.0)
    data.lower_target_panel_length = st.sidebar.number_input("Целевая длина панели нижних ферм, м", 0.4, 4.0, data.lower_target_panel_length, 0.1)
    data.target_panel_length = st.sidebar.number_input("Целевая длина панели верхних ферм, м", 0.4, 4.0, data.target_panel_length, 0.1)

    data.lower_count = lower_count
    data.upper_count = upper_count
    data.lower_height_mm = lower_height
    data.upper_height_mm = upper_height
    support_options = [SupportMode.EXISTING_10.value, SupportMode.LOWER_TRUSS_ENDS.value]
    default_support = data.support_mode.value if data.support_mode.value in support_options else SupportMode.LOWER_TRUSS_ENDS.value
    data.support_mode = SupportMode(
        st.sidebar.selectbox(
            "Опоры",
            support_options,
            support_options.index(default_support),
            help="EXISTING_10 — исходные 10 стоек. LOWER_TRUSS_ENDS — по две стойки на каждую нижнюю ферму.",
        )
    )
    if preset_name != "original" or lower_count != 4:
        data.lower_positions_x = None
        if data.support_mode == SupportMode.EXISTING_10:
            st.sidebar.warning("EXISTING_10 корректен только для исходных 4 нижних ферм; включён LOWER_TRUSS_ENDS.")
            data.support_mode = SupportMode.LOWER_TRUSS_ENDS

    data.lower_chord_section = st.sidebar.selectbox("Пояса нижних ферм", sections, sections.index(data.lower_chord_section))
    data.lower_web_section = st.sidebar.selectbox("Решётка нижних ферм", sections, sections.index(data.lower_web_section))
    data.upper_chord_section = st.sidebar.selectbox("Пояса верхних ферм", sections, sections.index(data.upper_chord_section))
    data.upper_web_section = st.sidebar.selectbox("Решётка верхних ферм", sections, sections.index(data.upper_web_section))
    data.column_section = st.sidebar.selectbox("Стойки", sections, sections.index(data.column_section))
    web_options = [item.value for item in WebType]
    data.lower_web_type = WebType(st.sidebar.selectbox("Заполнение нижних ферм", web_options, web_options.index(str(data.lower_web_type))))
    data.upper_web_type = WebType(st.sidebar.selectbox("Заполнение верхних ферм", web_options, web_options.index(str(data.upper_web_type))))

    st.sidebar.subheader("Высотные отметки")
    data.front_left_height = st.sidebar.number_input("Передний левый угол, м", 0.0, 10.0, data.front_left_height, 0.1)
    data.front_right_height = st.sidebar.number_input("Передний правый угол, м", 0.0, 10.0, data.front_right_height, 0.1)
    data.rear_left_height = st.sidebar.number_input("Задний левый угол, м", 0.0, 10.0, data.rear_left_height, 0.1)
    if data.front_left_height == data.front_right_height == data.rear_left_height:
        st.sidebar.warning("Высотные отметки кровли не заданы. Расчёт коэффициента формы снеговой нагрузки требует фактического уклона.")

    st.sidebar.subheader("Нагрузки")
    data.roof_dead_load_kpa = st.sidebar.number_input("Постоянная нагрузка, кН/м²", 0.0, 5.0, data.roof_dead_load_kpa, 0.01)
    data.snow_sg_kpa = st.sidebar.number_input("Sg снег на земле, кН/м²", 0.0, 10.0, data.snow_sg_kpa, 0.01)
    data.snow_ce = st.sidebar.number_input("ce", 0.0, 2.0, data.snow_ce, 0.05)
    data.snow_ct = st.sidebar.number_input("ct", 0.0, 2.0, data.snow_ct, 0.05)
    data.snow_gamma_f = st.sidebar.number_input("gamma_f", 1.0, 2.0, data.snow_gamma_f, 0.05)

    st.sidebar.subheader("Стоимость")
    data.steel_price_per_kg = st.sidebar.number_input("Сталь, руб/кг", 0.0, 1000.0, data.steel_price_per_kg, 5.0)
    data.fabrication_per_truss = st.sidebar.number_input("Изготовление фермы, руб", 0.0, 1_000_000.0, data.fabrication_per_truss, 1000.0)
    data.column_cost = st.sidebar.number_input("Стойка, руб", 0.0, 1_000_000.0, data.column_cost, 1000.0)
    data.foundation_cost = st.sidebar.number_input("Фундамент, руб", 0.0, 1_000_000.0, data.foundation_cost, 1000.0)
    data.connection_cost = st.sidebar.number_input("Монтажный узел, руб", 0.0, 100_000.0, data.connection_cost, 100.0)
    data.base_fixity = BaseFixity(st.sidebar.selectbox("Закрепление оснований", [item.value for item in BaseFixity]))
    return data


def naves_one_inputs(data: CanopyInput, sections: list[str]) -> CanopyInput:
    st.sidebar.caption("NavesOne: вариант по листам 2–5 «Навес 1809», 5 ферм В, 10 ферм Ф, 12 стоек — без K6.")
    st.sidebar.subheader("Высоты ферм")
    data.lower_height_mm = st.sidebar.number_input(
        "Нижние фермы В, мм", 50.0, 1200.0, float(data.lower_height_mm), 10.0,
        key="naves_lower_height_mm",
    )
    data.upper_height_mm = st.sidebar.number_input(
        "Верхние фермы F со стороны K5, мм", 50.0, 1200.0, float(data.upper_height_mm), 10.0,
        key="naves_upper_height_mm",
    )
    data.upper_end_height_mm = st.sidebar.number_input(
        "F1 на противоположном торце, мм", 50.0, 1200.0,
        float(data.upper_end_height_mm if data.upper_end_height_mm is not None else data.upper_height_mm),
        10.0, key="naves_upper_end_height_mm",
    )
    st.sidebar.caption(
        "Высоты противоположных торцов F2–F10 рассчитываются по их длине с тем же уклоном поясов, "
        "что у F1. Общие плоскости поясов сохраняются; отметки кровли меняются."
    )
    if data.upper_max_diagonal_mm is not None:
        st.sidebar.caption(
            f"Укосины F: не более {data.upper_max_diagonal_mm:g} мм между узлами. "
            "Шаг решётки подбирается автоматически; высоты F должны быть меньше этого предела."
        )
    if data.column_height_mm is not None:
        st.sidebar.caption(
            f"Исходная отметка опирания нижних ферм: {data.column_height_mm:g} мм. "
            "Стойки front/rear укорочены под перекладины 100×100×3 мм."
        )
    st.sidebar.dataframe(
        pd.DataFrame(upper_truss_dimensions(data)),
        hide_index=True, width="stretch", height=388,
        column_config={
            "L, м": st.column_config.NumberColumn(format="%.3f", help="Длина в плане, включая свес 725 мм."),
            "K5, мм": st.column_config.NumberColumn(format="%.1f", help="Высота торца со стороны K5."),
            "Торец, мм": st.column_config.NumberColumn(format="%.1f", help="Высота противоположного торца."),
        },
    )
    st.sidebar.subheader("Сечения")
    for attr, label in (
        ("lower_chord_section", "Пояса ферм В"),
        ("lower_web_section", "Решётка ферм В"),
        ("upper_chord_section", "Пояса ферм Ф"),
        ("upper_web_section", "Решётка ферм Ф"),
        ("column_section", "Стойки"),
    ):
        current = getattr(data, attr)
        setattr(data, attr, st.sidebar.selectbox(label, sections, sections.index(current)))
    st.sidebar.subheader("Нагрузки")
    data.roof_dead_load_kpa = st.sidebar.number_input("Постоянная нагрузка, кН/м²", 0.0, 5.0, data.roof_dead_load_kpa, 0.01)
    data.snow_sg_kpa = st.sidebar.number_input("Sg снег на земле, кН/м²", 0.0, 10.0, data.snow_sg_kpa, 0.01)
    data.snow_ce = st.sidebar.number_input("ce", 0.0, 2.0, data.snow_ce, 0.05)
    data.snow_ct = st.sidebar.number_input("ct", 0.0, 2.0, data.snow_ct, 0.05)
    data.snow_gamma_f = st.sidebar.number_input("gamma_f", 1.0, 2.0, data.snow_gamma_f, 0.05)
    _, _, snow_pressure_kpa = snow_design_pressure_kpa(
        data.snow_sg_kpa, data.snow_ce, data.snow_ct,
        roof_plane_from_input(data), data.snow_gamma_f, data.snow_mu,
    )
    snow_total_kn = snow_pressure_kpa * naves_one_roof_area()
    st.sidebar.metric(
        "Общий вес снега", f"{snow_total_kn / G_ACCEL:.2f} т",
        help="Эквивалентная масса расчётной снеговой нагрузки на всю кровлю, с учётом gamma_f. Без веса конструкции и кровли.",
    )
    st.sidebar.caption(f"{snow_total_kn:.2f} кН = {snow_pressure_kpa:.3f} кН/м² × {naves_one_roof_area():.2f} м²")
    st.sidebar.subheader("Расчётные допущения")
    data.base_fixity = BaseFixity(st.sidebar.selectbox("Закрепление оснований", [item.value for item in BaseFixity]))
    data.steel_price_per_kg = st.sidebar.number_input("Сталь, руб/кг", 0.0, 1000.0, data.steel_price_per_kg, 5.0)
    st.sidebar.warning("Схема предварительная: горизонтальные перемещения узлов закреплены в МКЭ; соединения, основания и свойства кровли требуют уточнения.")
    return data


def geometry_summary(model, data: CanopyInput) -> None:
    masses = mass_breakdown(model)
    c1, c2, c3 = st.columns(3)
    c1.metric("Узлы", len(model.nodes))
    c2.metric("Элементы", len(model.elements))
    c3.metric("Стойки", len(model.columns))
    if data.preset_name == "naves_one":
        for warning in model.warnings:
            st.caption(warning)
    else:
        st.caption(f"Original supports: {ORIGINAL_SUPPORTS_BY_TRUSS}")
    st.caption(f"Масса до расчёта: нижние {masses['lower']:.1f} кг, верхние {masses['upper']:.1f} кг, стойки {masses['columns']:.1f} кг, перекладины {masses['crossbeams']:.1f} кг.")


def show_results(data: CanopyInput, model, result) -> None:
    if not result.ok:
        st.error(result.message)
        return
    cost = cost_breakdown(data, model)
    max_util = float(result.elements["utilization"].max())
    max_reaction = float(result.reactions["Rz_kN"].abs().max())
    cols = st.columns(7)
    cols[0].metric("Площадь", f"{model.loaded_roof_area:.1f} м²")
    cols[1].metric("Масса", f"{mass_breakdown(model)['total']:.0f} кг")
    cols[2].metric("Max Uz", f"{result.max_uz_mm:.1f} мм")
    cols[3].metric("Max |N|", f"{result.max_abs_n_kn:.1f} кН")
    cols[4].metric("Max Rz", f"{max_reaction:.1f} кН")
    cols[5].metric(
        "Max util (осевое)", f"{max_util:.2f}",
        help="max |N|/(A·Ry): только осевые напряжения. Устойчивость, изгиб и соединения не проверяются.",
    )
    cols[6].metric("Стоимость", f"{cost['total_cost']:,.0f} руб")
    st.caption(
        "Max util = max |N|/(A·Ry), Ry = "
        f"{data.steel_resistance_mpa:g} МПа. Это не полная проверка несущей способности. "
        "В текущей модели горизонтальные перемещения всех узлов закреплены: "
        "такие закрепления воспринимают горизонтальные усилия и меняют работу ферм. "
        "Баланс вертикальных нагрузок не подтверждает корректность этой расчётной схемы."
    )

    lb = result.load_breakdown
    st.info(
        f"Sg={lb.sg_kpa:.2f}, ce={lb.ce:.2f}, ct={lb.ct:.2f}, mu={lb.mu:.3f}, "
        f"gamma_f={lb.gamma_f:.2f}, S0={lb.s0_kpa:.3f} кН/м², Sd={lb.sd_kpa:.3f} кН/м²."
    )
    balance_fields = (
        "snow_load_total_kN", "roof_dead_load_total_kN",
        "lower_trusses_self_weight_kN", "upper_trusses_self_weight_kN",
        "columns_self_weight_kN", "crossbeams_self_weight_kN", "total_self_weight_kN",
        "total_applied_vertical_load_kN", "sum_vertical_reactions_kN",
    )
    st.subheader("Баланс вертикальной нагрузки, кН")
    st.dataframe(pd.DataFrame({"Показатель": balance_fields, "кН": [getattr(lb, name) for name in balance_fields]}), hide_index=True)
    st.caption(f"Допуск баланса: {VERTICAL_BALANCE_TOLERANCE_KN:g} кН")

    if st.button("Сохранить вариант для сравнения"):
        st.session_state["variants"].append(variant_summary(f"Variant {len(st.session_state['variants']) + 1}", data, model, result))

    tab_names = ["Элементы", "Узлы", "Реакции", "Нагрузки", "Top utilization", "Экспорт"]
    tabs = st.tabs(tab_names)
    tabs[0].dataframe(result.elements.sort_values("utilization", ascending=False), use_container_width=True)
    tabs[1].dataframe(result.nodes, use_container_width=True)
    tabs[2].dataframe(result.reactions.sort_values("Rz_kN", ascending=False), use_container_width=True)
    tabs[3].dataframe(result.loads, use_container_width=True)
    tabs[4].dataframe(top_utilized(result.elements), use_container_width=True)
    with tabs[5]:
        export_payload = {
            "input": asdict(data),
            "summary": variant_summary("current", data, model, result),
            "loads": asdict(result.load_breakdown),
        }
        st.download_button("Export JSON", json.dumps(export_payload, ensure_ascii=False, indent=2, default=str), "naves-calc.json")
        st.download_button("Export elements CSV", result.elements.to_csv(index=False), "elements.csv")
        st.download_button("Export results.xlsx", results_xlsx(result), "results.xlsx")
        st.download_button("Export HTML report", html_report(data, result), "report.html")


def comparison_panel() -> None:
    st.header("Сравнение схем")
    variants = st.session_state.get("variants", [])
    if not variants:
        st.caption("Сохраните рассчитанный вариант, чтобы сравнить его с другими.")
        return
    table = pd.DataFrame([{key: value for key, value in item.items() if key != "input"} for item in variants])
    st.dataframe(table, use_container_width=True)


def apply_camera(fig, camera_name: str, projection: str = "perspective") -> None:
    cameras = {
        "Изометрия": dict(eye=dict(x=1.6, y=-1.8, z=1.2)),
        "План": dict(eye=dict(x=0.0, y=0.0, z=2.6)),
        "Спереди": dict(eye=dict(x=0.0, y=-2.4, z=0.6)),
        "Сзади": dict(eye=dict(x=0.0, y=2.4, z=0.6)),
        "Слева": dict(eye=dict(x=-2.4, y=0.0, z=0.6)),
        "Справа": dict(eye=dict(x=2.4, y=0.0, z=0.6)),
    }
    camera = cameras[camera_name]
    camera["projection"] = dict(type=projection)
    if projection == "orthographic" and camera_name not in ("Изометрия", "План"):
        camera["eye"]["z"] = 0.0
    if camera_name == "План":
        camera["up"] = dict(x=0.0, y=1.0, z=0.0)
    fig.update_layout(scene_camera=camera)


if __name__ == "__main__":
    main()
