from src.models import BaseFixity, CanopyInput, Element, StructuralModel


MODEL_LIMITATIONS = (
    "Линейная упругая вертикальная идеализация: горизонтальные перемещения исходных узлов "
    "закреплены; во многих узлах закреплены также повороты. Эти связи воспринимают усилия. "
    "Добавленные узлы сетки свободны. Типы элементов определяются генератором геометрии. "
    "Входные параметры типа соединений, типа фермы и числа уровней пока не переключают расчётную схему. "
    "Осевой коэффициент дополнен проверкой упругих нормальных напряжений N–M; устойчивость, полная "
    "прочность, соединения и эффекты второго порядка не проверяются."
)


def model_limitations(data):
    if data.spatial_frame:
        return (
            "Пространственная линейная упругая рама. Стойки, перекладины, пояса и решётка "
            "работают на растяжение/сжатие, изгиб и кручение. Сварные соединения приняты "
            "абсолютно жёсткими; их фактическая жёсткость и прочность не подтверждены. "
            "Жёсткие вставки учитывают смещения осей у перекладин и торцов B; "
            "оси поясов в пересечениях ферм идеализированы совпадающими. "
            "Закреплены только основания стоек. Мембрана и её основание не считаются диском жёсткости. "
            "Решётка не дробится; собственный вес приложен к концам элементов. "
            "Устойчивость, второй порядок, срез, кручение по прочности и соединения требуют отдельных проверок."
        )
    if not data.spatial_diagnostic:
        return MODEL_LIMITATIONS
    return (
        "Диагностика: искусственные поступательные закрепления и повороты поясов сняты. "
        "Закреплены основания и неработающие вращательные степени свободы чисто стержневых узлов. "
        "Существующие связи опирания передают только Uz; пространственные связи, эксцентриситеты "
        "и реальные соединения не восстановлены. Возможны механизмы, включая свободное кручение поясов. "
        "Успешное линейное решение не подтверждает устойчивость и несущую способность."
    )


def node_restraints(data: CanopyInput, model: StructuralModel) -> dict[int, tuple[int, ...]]:
    """Actual restraints of the current vertical idealization, not physical joint releases."""
    supports = set(model.support_node_ids)
    if data.spatial_frame:
        rotation = int(data.base_fixity != BaseFixity.PINNED)
        return {n: (1, 1, 1, rotation, rotation, rotation) for n in supports}
    if data.spatial_diagnostic:
        # Remove all artificial translational supports and chord rotation clamps.
        # Only rotations with no beam stiffness at all are suppressed.
        beam_nodes = {n for e in model.elements if e.is_beam or e.role in ("column", "crossbeam")
                      for n in (e.i, e.j)}
        return {n: ((1, 1, 1, *([int(data.base_fixity != BaseFixity.PINNED)] * 3))
                    if n in supports else (0, 0, 0, 1, 1, 1))
                for n in model.nodes if n in supports or n not in beam_nodes}
    chord_nodes = {n for e in model.elements if e.is_beam for n in (e.i, e.j)}
    web_nodes = {n for e in model.elements if e.role == "lower_web" for n in (e.i, e.j)}
    chord_bearing_nodes = chord_nodes - web_nodes
    upper_beam_nodes = {n for e in model.elements if e.is_beam and e.role == "upper_chord" for n in (e.i, e.j)}
    # Each isolated straight upper chord has an otherwise free rigid torsional
    # rotation. Set one torsional datum per chain. Upper chords lie in XZ planes:
    # fixing Rx removes that null mode while Ry (in-plane bending) stays free.
    torsion_references = {row[0] for rows in (model.upper_trusses, model.upper_bottom_trusses)
                          for row in rows.values() if row and row[0] in upper_beam_nodes}
    restraints = {}
    for node_id in model.nodes:
        if node_id in supports:
            rotation = int(data.base_fixity != BaseFixity.PINNED)
            restraints[node_id] = (1, 1, 1, rotation, rotation, rotation)
        elif node_id not in model.mesh_node_ids:
            rotation = int(node_id not in chord_bearing_nodes)
            restraints[node_id] = (1, 1, 0, rotation, rotation, rotation)
            if node_id in torsion_references:
                restraints[node_id] = (1, 1, 0, 1, 0, 0)
    return restraints


def beam_width_axis(model: StructuralModel, element: Element) -> tuple[float, float, float]:
    """OpenSees vecxz along section width: local z = width, local y = depth.

    Consequently Iy receives RHS.iy (b³h) and Iz receives RHS.ix (bh³).
    Matches the solid view: columns have width along global X; horizontal
    members have depth in the vertical plane through the member axis.
    """
    a, b = model.nodes[element.i], model.nodes[element.j]
    dx, dy = b.x - a.x, b.y - a.y
    if dx == 0 and dy == 0:
        return (1.0, 0.0, 0.0)
    return (-dy, dx, 0.0)


def uses_beam(data: CanopyInput, element: Element) -> bool:
    """Welded-frame hypothesis gives webs flexural and torsional stiffness too."""
    return data.spatial_frame or element.is_beam or element.role in ("column", "crossbeam")


def apply_bearing_constraints(ops, data: CanopyInput, model: StructuralModel):
    """Never replace eccentric rigid joints with equal translations only."""
    if data.spatial_frame:
        retained = {a for a, _ in model.bearing_links}
        constrained = [b for _, b in model.bearing_links]
        if retained.intersection(constrained) or len(constrained) != len(set(constrained)):
            raise ValueError("Rigid joints cannot have chained constraints or duplicate slave nodes")
        if set(constrained).intersection(model.support_node_ids):
            raise ValueError("A base support cannot also be a rigid-link slave")
        for a, b in model.bearing_links:
            ops.rigidLink("beam", a, b)
    else:
        for a, b in model.bearing_links:
            ops.equalDOF(a, b, 3)
