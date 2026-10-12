"""Preliminary bar model transcribed from the 15-sheet 'Навес 1809' drawing.

The DXF contains triangulated surfaces and drafting views, not member axes.
Coordinates below are taken from sheets 2-5; all dimensions are metres.
"""

from __future__ import annotations

from math import ceil, dist, isclose

from src.geometry.canopy import roof_plane_from_input
from src.geometry.truss import ElementRegistry, NodeRegistry, add_web_pattern, validate_connectivity
from src.models import CanopyInput, Column, StructuralModel, WebType
from src.sections.database import section_database


WIDTH = 12.0
UPPER_OVERHANG = 0.725
LEFT_REAR_X = 2.588
LEFT_REAR_Y = 9.659
RIGHT_REAR_Y = 7.030
FRONT_XS = (0.0, 2.588, 6.0, 9.0, 12.0)
FRONT_COLUMN_NAMES = ("K7", "K1-front", "K2-front", "K3-front", "K4-front")
REAR_XS = (2.588, 6.0, 9.0, 12.0)
REAR_YS = (9.659, 8.725, 7.878, 7.030)
REAR_COLUMN_NAMES = ("K1-rear", "K2-rear", "K3-rear", "K4-rear")
UPPER_YS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.848, 8.696)
OUTLINE = [(0.0, 0.0), (WIDTH, 0.0), (WIDTH, RIGHT_REAR_Y), (LEFT_REAR_X, LEFT_REAR_Y)]
CROSSBEAM_SECTION = "100x100x3"


def left_x(y: float) -> float:
    return LEFT_REAR_X * y / LEFT_REAR_Y


def right_x(y: float) -> float:
    if y <= RIGHT_REAR_Y:
        return WIDTH
    return WIDTH + (LEFT_REAR_X - WIDTH) * (y - RIGHT_REAR_Y) / (LEFT_REAR_Y - RIGHT_REAR_Y)


def roof_area() -> float:
    return _area(OUTLINE)


def upper_height_at(data: CanopyInput, x: float, y: float) -> float:
    """Vertical F-truss depth in metres, with F1's taper shared by all spans."""
    start_height = data.upper_height_mm / 1000.0
    end_height = (data.upper_end_height_mm / 1000.0
                  if data.upper_end_height_mm is not None else start_height)
    start_x = left_x(y) - UPPER_OVERHANG
    reference_span = right_x(UPPER_YS[0]) - (left_x(UPPER_YS[0]) - UPPER_OVERHANG)
    return start_height + (x - start_x) / reference_span * (end_height - start_height)


def upper_truss_dimensions(data: CanopyInput) -> list[dict]:
    """Unrounded plan lengths and end depths for the sidebar schedule."""
    return [
        {"Ферма": f"F{index + 1}",
         "L, м": right_x(y) - (left_x(y) - UPPER_OVERHANG),
         "K5, мм": data.upper_height_mm,
         "Торец, мм": upper_height_at(data, right_x(y), y) * 1000.0}
        for index, y in enumerate(UPPER_YS)
    ]


def _area(polygon: list[tuple[float, float]]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]))) / 2.0


def _clip(polygon: list[tuple[float, float]], axis: int, limit: float, keep_greater: bool) -> list[tuple[float, float]]:
    if not polygon:
        return []
    result: list[tuple[float, float]] = []
    for p, q in zip(polygon[-1:] + polygon[:-1], polygon):
        inside_p = p[axis] >= limit if keep_greater else p[axis] <= limit
        inside_q = q[axis] >= limit if keep_greater else q[axis] <= limit
        if inside_p != inside_q:
            t = (limit - p[axis]) / (q[axis] - p[axis])
            result.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        if inside_q:
            result.append(q)
    return result


def _cell_area(x0: float, x1: float, y0: float, y1: float) -> float:
    polygon = OUTLINE
    for axis, limit, greater in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        polygon = _clip(polygon, axis, limit, greater)
    return _area(polygon) if len(polygon) >= 3 else 0.0


def _stations(start: float, end: float, extras: list[float]) -> list[float]:
    values = [start, end, *extras]
    values += [float(i) for i in range(int(start) + 1, int(end) + 1) if start < i < end]
    return sorted({round(value, 8) for value in values if start - 1e-8 <= value <= end + 1e-8})


def _subdivide_upper_panels(data: CanopyInput, y: float, xs: list[float]) -> list[float]:
    """Retain all support stations and limit both possible panel diagonals."""
    if data.upper_max_diagonal_mm is None:
        return xs
    limit = data.upper_max_diagonal_mm / 1000.0
    roof = roof_plane_from_input(data)
    if limit <= max(upper_height_at(data, x, y) for x in xs):
        raise ValueError(
            f"Для укосин F не длиннее {data.upper_max_diagonal_mm:g} мм "
            "высоты верхних ферм должны быть меньше этого значения."
        )

    def split(a: float, b: float) -> list[float]:
        top_a, top_b = (a, y, roof.z(a, y)), (b, y, roof.z(b, y))
        bottom_a = (a, y, top_a[2] - upper_height_at(data, a, y))
        bottom_b = (b, y, top_b[2] - upper_height_at(data, b, y))
        if max(dist(top_a, bottom_b), dist(bottom_a, top_b)) <= limit - 1e-8:
            return [b]
        mid = round((a + b) / 2, 8)
        if mid - a < 1e-6 or b - mid < 1e-6:
            raise ValueError("Невозможно подобрать шаг решётки F: уменьшите высоту верхних ферм.")
        return split(a, mid) + split(mid, b)

    return [xs[0], *[x for a, b in zip(xs, xs[1:]) for x in split(a, b)]]


def generate_naves_one(data: CanopyInput) -> StructuralModel:
    """Build the sheet geometry while retaining editable loads and sections."""
    roof = roof_plane_from_input(data)
    nodes = NodeRegistry()
    elements = ElementRegistry()
    lower_trusses: dict[str, list[int]] = {}
    upper_trusses: dict[str, list[int]] = {}
    upper_bottom_trusses: dict[str, list[int]] = {}
    lower_top_at: dict[tuple[int, float], int] = {}
    lower_bottom_at: dict[tuple[int, float], int] = {}
    lower_height = data.lower_height_mm / 1000.0
    upper_start_height = data.upper_height_mm / 1000.0
    upper_end_height = (
        data.upper_end_height_mm / 1000.0
        if data.upper_end_height_mm is not None else upper_start_height
    )
    if min(lower_height, upper_start_height, upper_end_height) <= 0:
        raise ValueError("Truss heights must be positive")

    supports = [(name, x, 0.0) for name, x in zip(FRONT_COLUMN_NAMES, FRONT_XS)]
    supports += [(name, x, y) for name, x, y in zip(REAR_COLUMN_NAMES, REAR_XS, REAR_YS)]
    supports += [("K5", left_x(4.829), 4.829)]
    supports += [("K4-middle-1", WIDTH, 2.659), ("K4-middle-2", WIDTH, 4.829)]

    # B1 follows the sloping left edge. B2-B5 are straight longitudinal trusses.
    for index, (front_x, end_y) in enumerate(zip(FRONT_XS, (LEFT_REAR_Y, *REAR_YS))):
        name = f"B{index + 1}"
        support_ys = [y for _, x, y in supports if (isclose(x, left_x(y), abs_tol=1e-7) if index == 0 else isclose(x, front_x, abs_tol=1e-7))]
        # Bearings split the chords only; neither columns nor F trusses alter the web.
        plan_length = dist((front_x, 0.0), (left_x(end_y) if index == 0 else front_x, end_y))
        panel_count = max(1, ceil(plan_length / max(data.lower_target_panel_length, 0.1)))
        web_ys = [round(end_y * i / panel_count, 8) for i in range(panel_count + 1)]
        top_ys = sorted(set(web_ys) | {round(y, 8) for y in UPPER_YS if y <= end_y})
        bottom_ys = sorted(set(web_ys) | {round(y, 8) for y in support_ys})
        top_nodes: list[int] = []
        bottom_nodes: list[int] = []
        for y in top_ys:
            x = left_x(y) if index == 0 else front_x
            top_z = roof.z(x, y) - upper_height_at(data, x, y)
            top_nodes.append(nodes.get(x, y, top_z, f"{name}-top-{y:.3f}", "lower_top"))
            lower_top_at[(index, y)] = top_nodes[-1]
        for y in bottom_ys:
            x = left_x(y) if index == 0 else front_x
            top_z = roof.z(x, y) - upper_height_at(data, x, y)
            bottom_nodes.append(nodes.get(x, y, top_z - lower_height, f"{name}-bottom-{y:.3f}", "lower_bottom"))
            lower_bottom_at[(index, y)] = bottom_nodes[-1]
        for row, label in ((top_nodes, "top"), (bottom_nodes, "bottom")):
            for a, b in zip(row, row[1:]):
                elements.add(a, b, "lower_chord", name, data.lower_chord_section,
                             f"{name} {label} chord", is_beam=True)
        web_top_nodes = [lower_top_at[(index, y)] for y in web_ys]
        web_bottom_nodes = [lower_bottom_at[(index, y)] for y in web_ys]
        add_web_pattern(elements, web_top_nodes, web_bottom_nodes, "lower_web", name, data.lower_web_section, WebType(data.lower_web_type))
        lower_trusses[name] = top_nodes

    roof_node_ids: list[int] = []
    bearing_links: list[tuple[int, int]] = []
    tributary_areas: dict[int, float] = {}
    for row_index, y in enumerate(UPPER_YS):
        name = f"F{row_index + 1}"
        start_x = left_x(y) - UPPER_OVERHANG  # overhang shown on sheet 4
        end_x = right_x(y)
        crossings = [left_x(y), *[x for x, end_y in zip(REAR_XS, REAR_YS) if y <= end_y + 1e-8]]
        xs = _subdivide_upper_panels(data, y, _stations(start_x, end_x, crossings))
        top_nodes: list[int] = []
        bottom_nodes: list[int] = []
        for x in xs:
            top_z = roof.z(x, y)
            top_nodes.append(nodes.get(x, y, top_z, f"{name}-roof-{x:.3f}", "roof"))
            match = next((i for i in range(5) if (isclose(x, left_x(y), abs_tol=1e-7) if i == 0 else isclose(x, FRONT_XS[i], abs_tol=1e-7)) and y <= (LEFT_REAR_Y, *REAR_YS)[i] + 1e-8), None)
            if match is None:
                bottom_nodes.append(nodes.get(x, y, top_z - upper_height_at(data, x, y), f"{name}-bottom-{x:.3f}", "upper_bottom"))
            else:
                # Transfer vertical load without rigidly joining the rotations
                # of orthogonal continuous chords or inheriting a lower-web clamp.
                retained = lower_top_at[(match, round(y, 8))]
                bearing = nodes.independent(*nodes.nodes[retained].coords,
                                            f"{name}-bearing-{x:.3f}", "upper_bottom")
                bottom_nodes.append(bearing)
                bearing_links.append((retained, bearing))
        for row, label in ((top_nodes, "top"), (bottom_nodes, "bottom")):
            for a, b in zip(row, row[1:]):
                elements.add(a, b, "upper_chord", name, data.upper_chord_section,
                             f"{name} {label} chord", is_beam=True)
        add_web_pattern(elements, top_nodes, bottom_nodes, "upper_web", name, data.upper_web_section, WebType(data.upper_web_type))
        upper_trusses[name] = top_nodes
        upper_bottom_trusses[name] = bottom_nodes
        roof_node_ids.extend(top_nodes)

        y0 = 0.0 if row_index == 0 else (UPPER_YS[row_index - 1] + y) / 2.0
        y1 = LEFT_REAR_Y if row_index == len(UPPER_YS) - 1 else (y + UPPER_YS[row_index + 1]) / 2.0
        for i, node_id in enumerate(top_nodes):
            x0 = -1.0 if i == 0 else (xs[i - 1] + xs[i]) / 2.0
            x1 = WIDTH + 1.0 if i == len(xs) - 1 else (xs[i] + xs[i + 1]) / 2.0
            tributary_areas[node_id] = _cell_area(x0, x1, y0, y1)

    columns: list[Column] = []
    support_node_ids: list[int] = []
    beam_rows: dict[str, list[int]] = {"front": [], "rear": []}
    sections = section_database()
    beam_height = sections[CROSSBEAM_SECTION].height
    lower_half_height = sections[data.lower_chord_section].height / 2.0
    for name, x, y in supports:
        match = next(i for i in range(5) if (isclose(x, left_x(y), abs_tol=1e-7) if i == 0 else isclose(x, FRONT_XS[i], abs_tol=1e-7)) and y <= (LEFT_REAR_Y, *REAR_YS)[i] + 1e-8)
        top_id = lower_bottom_at[(match, round(y, 8))]
        row = "front" if name in FRONT_COLUMN_NAMES else "rear" if name in REAR_COLUMN_NAMES else None
        if row is not None:
            truss_id = top_id
            truss_z = nodes.nodes[truss_id].z
            beam_id = nodes.get(x, y, truss_z - lower_half_height - beam_height / 2.0,
                                f"{row}-{name}-axis", "crossbeam")
            top_z = truss_z - lower_half_height - beam_height
            if top_z <= 0:
                raise ValueError("Недостаточная высота стойки для установки перекладины.")
            top_id = nodes.get(x, y, top_z, f"{name}-top", "column_top")
            beam_rows[row].append(beam_id)
            bearing_links.extend([(beam_id, top_id), (beam_id, truss_id)])
        nodes.nodes[top_id].roles.add("column_top")
        base_id = nodes.get(x, y, 0.0, f"{name}-base", "support")
        elements.add(base_id, top_id, "column", name, data.column_section, name)
        columns.append(Column(name, x, y, base_id, top_id))
        support_node_ids.append(base_id)

    for row, beam_nodes in beam_rows.items():
        for a, b in zip(beam_nodes, beam_nodes[1:]):
            elements.add(a, b, "crossbeam", row, CROSSBEAM_SECTION, f"Перекладина {row}")

    model = StructuralModel(
        nodes=nodes.nodes, elements=elements.elements, columns=columns,
        lower_trusses=lower_trusses, upper_trusses=upper_trusses,
        upper_bottom_trusses=upper_bottom_trusses, roof_node_ids=roof_node_ids,
        support_node_ids=support_node_ids, loaded_roof_area=roof_area(),
        roof_outline=OUTLINE, roof_tributary_areas=tributary_areas,
        additional_dead_mass_kg=105.64 + 145.67,
        bearing_links=bearing_links,
        warnings=[
            "Изменённая схема: 12 стоек, K6 на оси 1–Б исключена из исходного чертежа.",
            "Перекладины front (включая K7) и rear: 100×100×3 мм. "
            f"Отметки ферм сохранены; стойки под перекладинами укорочены на {(beam_height + lower_half_height) * 1000:g} мм.",
            f"Нижние фермы В: {data.lower_height_mm:g} мм; верхние Ф: "
            f"F1 — {upper_start_height * 1000:g} → {upper_end_height * 1000:g} мм; "
            f"F2–F10 — от {upper_start_height * 1000:g} мм со стороны K5 с тем же уклоном поясов, что у F1. "
            + (f"Исходная отметка опирания нижних ферм {data.column_height_mm:g} мм; отметки кровли рассчитаны по высотам ферм."
               if data.column_height_mm is not None else "Отметки кровли сохранены, высоты стоек пересчитаны."),
            "Стержневая идеализация по листам 2–5 PDF; узлы и закрепления требуют подтверждения инженером.",
            "Связи 40×20×1,5 учтены собственным весом 251,31 кг по спецификации, без их жёсткости.",
            "В текущем МКЭ горизонтальные перемещения исходных узлов закреплены; результаты описывают только вертикальную идеализацию.",
            "Решётка нижних ферм имеет равномерный шаг независимо от стоек и верхних ферм. "
            "Узлы опирания добавлены только в пояса, без дополнительных стоек решётки. "
            "Оба пояса учтены балочными элементами "
            "с закреплёнными поворотами в узлах решётки; дополнительные узлы опирания "
            "не закрепляют поворот пояса.",
            "Оба пояса верхних ферм — непрерывные балочные цепочки с учётом изгиба и детализации МКЭ. "
            "Решётка остаётся шарнирной. Опирание на нижние фермы передаёт вертикальное перемещение "
            "без связи поворотов; нагрузка кровли приложена в исходных узлах. "
            "В начале каждого верхнего пояса задана одна фиксация Rx для исключения свободного "
            "вращения вокруг его оси; поворот Ry при изгибе в плоскости фермы свободен.",
            "Перекладины учтены балочными элементами с закреплёнными поворотами узлов; "
            "опирание передаёт вертикальные перемещения. Нормальные напряжения N–M проверяются; полная проверка прочности и соединений не выполняется.",
        ],
    )
    if data.spatial_diagnostic:
        from src.fem.assumptions import model_limitations
        model.warnings = model.warnings[:5] + [model_limitations(data)]
    if abs(sum(tributary_areas.values()) - model.loaded_roof_area) > 1e-7:
        raise ValueError("Roof tributary areas do not cover the drawing outline")
    validate_connectivity(model)
    from src.geometry.refinement import refine_beams

    return refine_beams(model, data.fem_subdivisions)
