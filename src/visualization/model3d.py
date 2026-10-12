from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from src.geometry.canopy import CanopyPlan, roof_plane_from_input
from src.loads.snow_sp20 import snow_design_pressure_kpa
from src.loads.scenarios import PATTERNS, snow_pattern_loads
from src.models import CanopyInput, StructuralModel
from src.sections.database import section_database
from src.sections.rhs import RHSSection


ROLE_COLORS = {
    "column": "#777777",
    "crossbeam": "#b8860b",
    "lower_chord": "#1f77b4",
    "lower_web": "#4f9fdf",
    "upper_chord": "#d62728",
    "upper_web": "#ff7f7f",
}


def model_figure(
    data: CanopyInput,
    model: StructuralModel,
    result=None,
    mode: str = "GEOMETRY",
    deformation_scale: float = 20.0,
    displacement_component: str = "UZ",
) -> go.Figure:
    fig = go.Figure()
    realistic = mode in ("REALISTIC", "REALISTIC_DEFORMED")
    if not realistic:
        _add_roof(fig, data, model)

    disp = {}
    if result is not None and result.ok and not result.nodes.empty:
        disp = {
            int(row.node): (row.Ux_mm / 1000.0, row.Uy_mm / 1000.0, row.Uz_mm / 1000.0)
            for row in result.nodes.itertuples()
        }
    element_values = {}
    if result is not None and result.ok and not result.elements.empty:
        if mode == "AXIAL_FORCE":
            element_values = dict(zip(result.elements["element"], result.elements["N_kN"]))
        elif mode == "UTILIZATION":
            element_values = dict(zip(result.elements["element"], result.elements["utilization"]))

    if realistic:
        deformed = mode == "REALISTIC_DEFORMED"
        _add_realistic_elements(fig, model, disp if deformed else {}, deformation_scale)
        if deformed:
            fig.update_layout(title=dict(
                text=(f"Деформированный каркас · масштаб ×{deformation_scale:g}" if disp
                      else "Исходный каркас · для деформаций выполните расчёт"),
                font=dict(size=15),
            ))
    elif mode == "DISPLACEMENT" and disp:
        _add_displacements(fig, model, disp, displacement_component)
    elif mode == "DEFORMED" and disp:
        _add_elements(fig, model, {}, "Исходная", opacity=0.25)
        _add_elements(fig, model, disp, "Деформированная", deformation_scale, opacity=1.0)
    elif element_values:
        _add_colored_elements(fig, model, element_values, mode)
    else:
        _add_elements(fig, model, {}, "Геометрия", opacity=0.3 if mode == "SNOW_LOAD" else 1.0)
        if mode == "DISPLACEMENT":
            fig.add_annotation(
                text="Для карты перемещений выполните успешный расчёт — нажмите «Рассчитать».",
                x=0.5, y=1.0, xref="paper", yref="paper", showarrow=False,
            )

    if not realistic:
        _add_supports(fig, model)
    fig.update_layout(
        scene=dict(
            xaxis_title="X, м",
            yaxis_title="Y, м",
            zaxis_title="Z, м",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="h"),
    )
    if mode == "SNOW_LOAD":
        _add_snow_load(fig, data, model)
    if realistic:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                bgcolor="#f2f4f6",
            ),
            paper_bgcolor="#f2f4f6", showlegend=False,
        )
    return fig


def _add_snow_load(fig: go.Figure, data: CanopyInput, model: StructuralModel) -> None:
    """Display the same applied snow forces as the solver, without solving FEM."""
    _, _, pressure = snow_design_pressure_kpa(
        data.snow_sg_kpa, data.snow_ce, data.snow_ct,
        roof_plane_from_input(data), data.snow_gamma_f, data.snow_mu,
    )
    loads = snow_pattern_loads(data, model, pressure)
    total_kn = abs(sum(loads.values())) / 1000.0
    fig.update_layout(
        title=dict(
            text=f"{PATTERNS[data.snow_pattern]}: q = {pressure:.3f} кН/м² · A кровли = {model.loaded_roof_area:.2f} м²"
                 f" · ΣF = {total_kn:.2f} кН",
            font=dict(size=15),
        ),
        margin=dict(t=65),
    )
    loaded = [(node_id, -force / 1000.0) for node_id, force in loads.items() if force < 0.0]
    if not loaded:
        fig.add_annotation(
            text="Снеговая нагрузка равна нулю", x=0.5, y=1.0,
            xref="paper", yref="paper", showarrow=False,
        )
        return

    max_force = max(force for _, force in loaded)
    # Arrow length is a visual scale in metres; forces remain in kN in hover/colorbar.
    span = max(node.x for node in model.nodes.values()) - min(node.x for node in model.nodes.values())
    max_length = max(span * 0.10, 0.5)
    xs, ys, zs, colors = [], [], [], []
    tips_x, tips_y, tips_z, forces, details = [], [], [], [], []
    for node_id, force in loaded:
        node = model.nodes[node_id]
        length = max_length * force / max_force
        xs.extend([node.x, node.x, None])
        ys.extend([node.y, node.y, None])
        zs.extend([node.z + length, node.z, None])
        colors.extend([force, force, 0.0])
        tips_x.append(node.x)
        tips_y.append(node.y)
        tips_z.append(node.z)
        forces.append(force)
        area = force / pressure
        details.append([node_id, area, force])

    scale = [[0, "#9ecae1"], [1, "#08519c"]]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", name="Снег: силы вниз",
        line=dict(color=colors, colorscale=scale, cmin=0, cmax=max_force, width=4),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Cone(
        x=tips_x, y=tips_y, z=tips_z,
        u=[0.0] * len(forces), v=[0.0] * len(forces),
        w=[-force / max_force for force in forces],
        anchor="tip", sizemode="absolute", sizeref=max_length * 0.22,
        colorscale=scale, cmin=0, cmax=1, showscale=False,
        name="Направление нагрузки", showlegend=False,
        customdata=details,
        hovertemplate="Узел %{customdata[0]}<br>A = %{customdata[1]:.3f} м²"
                      "<br>F↓ = %{customdata[2]:.3f} кН<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=tips_x, y=tips_y, z=tips_z, mode="markers",
        marker=dict(size=3, color=forces, colorscale=scale, cmin=0, cmax=max_force,
                    showscale=True, colorbar=dict(title="F↓, кН", thickness=15)),
        name="Узловая снеговая нагрузка", showlegend=False, customdata=details,
        hovertemplate="Узел %{customdata[0]}<br>A = %{customdata[1]:.3f} м²"
                      "<br>F↓ = %{customdata[2]:.3f} кН<extra></extra>",
    ))


def _rhs_mesh(start, end, section: RHSSection, width_axis) -> tuple[np.ndarray, np.ndarray]:
    """Hollow RHS with square-cut ends, in metres; height lies in the truss plane."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    axis /= np.linalg.norm(axis)
    u = np.asarray(width_axis, dtype=float)
    u = u - np.dot(u, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    vertices = []
    for inset in (0.0, section.thickness):
        w, h = section.width / 2 - inset, section.height / 2 - inset
        for origin in (start, end):
            vertices.extend(origin + x * u + y * v for x, y in ((w, h), (-w, h), (-w, -h), (w, -h)))
    triangles = []
    for i in range(4):
        j = (i + 1) % 4
        # Outer wall, inner wall, and the two annular end faces.
        for a, b, c, d in ((i, j, j + 4, i + 4), (i + 8, i + 12, j + 12, j + 8),
                           (i, i + 8, j + 8, j), (i + 4, j + 4, j + 12, i + 12)):
            triangles.extend(((a, b, c), (a, c, d)))
    return np.asarray(vertices), np.asarray(triangles)


def _add_realistic_elements(
    fig: go.Figure, model: StructuralModel, disp: dict | None = None, scale: float = 1.0,
) -> None:
    disp = disp or {}
    sections = section_database()
    vertices, triangles, labels = [], [], []
    for element in model.elements:
        if element.role == "column":
            width_axis = (1.0, 0.0, 0.0)
        elif element.role == "crossbeam":
            first, last = model.nodes[element.i], model.nodes[element.j]
            width_axis = (-(last.y - first.y), last.x - first.x, 0.0)
        else:
            trusses = model.lower_trusses if element.role.startswith("lower") else model.upper_trusses
            row = trusses[element.group]
            first, last = model.nodes[row[0]], model.nodes[row[-1]]
            width_axis = (-(last.y - first.y), last.x - first.x, 0.0)
        points, faces = _rhs_mesh(
            _node_xyz(model, element.i, disp, scale), _node_xyz(model, element.j, disp, scale),
            sections[element.section], width_axis,
        )
        triangles.extend(faces + len(vertices))
        vertices.extend(points)
        label = f"{_element_kind(element.role)} {element.group}<br>{element.section} мм"
        if disp:
            uz_i = disp.get(element.i, (0.0, 0.0, 0.0))[2] * 1000
            uz_j = disp.get(element.j, (0.0, 0.0, 0.0))[2] * 1000
            label += f"<br>Uz концов: {uz_i:.3f} / {uz_j:.3f} мм<br>Масштаб изображения: ×{scale:g}"
        labels.extend([label] * len(points))
    if not vertices:
        return
    xyz, faces = np.asarray(vertices), np.asarray(triangles)
    fig.add_trace(go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#9ca6b0", flatshading=True,
        lighting=dict(ambient=0.38, diffuse=0.8, specular=0.9, roughness=0.28, fresnel=0.25),
        lightposition=dict(x=1000, y=-1500, z=2000),
        hovertext=labels, hovertemplate="%{hovertext}<extra></extra>",
        name="Деформированный стальной каркас" if disp else "Стальной каркас", showlegend=False,
    ))


def _node_xyz(model: StructuralModel, node_id: int, disp: dict, scale: float) -> tuple[float, float, float]:
    node = model.nodes[node_id]
    dx, dy, dz = disp.get(node_id, (0.0, 0.0, 0.0))
    return node.x + dx * scale, node.y + dy * scale, node.z + dz * scale


def _element_kind(role: str) -> str:
    return {"column": "Стойка", "crossbeam": "Перекладина"}.get(role, "Ферма")


def _add_elements(fig: go.Figure, model: StructuralModel, disp: dict, name: str, scale: float = 1.0, opacity: float = 1.0) -> None:
    for role, color in ROLE_COLORS.items():
        xs, ys, zs, labels = [], [], [], []
        for element in model.elements:
            if element.role != role:
                continue
            p1 = _node_xyz(model, element.i, disp, scale)
            p2 = _node_xyz(model, element.j, disp, scale)
            xs += [p1[0], p2[0], None]
            ys += [p1[1], p2[1], None]
            zs += [p1[2], p2[2], None]
            label = f"{_element_kind(role)} {element.group}"
            labels += [label, label, None]
        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name=f"{name}: {role}",
                line=dict(color=color, width=5), opacity=opacity,
                hovertext=labels, hovertemplate="%{hovertext}<extra></extra>",
            ))


def _add_colored_elements(fig: go.Figure, model: StructuralModel, values: dict[int, float], mode: str) -> None:
    colorscale = "RdBu" if mode == "AXIAL_FORCE" else "Turbo"
    vmin = min(values.values())
    vmax = max(values.values())
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12
    for element in model.elements:
        p1 = _node_xyz(model, element.i, {}, 1.0)
        p2 = _node_xyz(model, element.j, {}, 1.0)
        value = values.get(element.id, 0.0)
        color = _value_color(value, vmin, vmax, colorscale)
        fig.add_trace(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                showlegend=False,
                line=dict(color=color, width=6),
                hovertext=(
                    f"{_element_kind(element.role)} {element.group}<br>"
                    + (f"N = {value:.3f} кН" if mode == "AXIAL_FORCE" else f"Util = {value:.3f}")
                ),
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )


def _add_displacements(fig: go.Figure, model: StructuralModel, disp: dict, component: str) -> None:
    if component not in ("UZ", "TOTAL"):
        raise ValueError(f"Unknown displacement component: {component}")
    node_ids = [node_id for node_id in model.nodes if node_id in disp]
    vectors = {node_id: np.asarray(disp[node_id]) * 1000.0 for node_id in node_ids}
    totals = {node_id: float(np.linalg.norm(vector)) for node_id, vector in vectors.items()}
    values = {node_id: abs(vector[2]) if component == "UZ" else totals[node_id]
              for node_id, vector in vectors.items()}
    maximum_node = max(values, key=values.get)
    maximum = values[maximum_node]
    label = "|Uz|" if component == "UZ" else "|U|"
    color_range = dict(colorscale=[[0, "#2166ac"], [0.35, "#1a9850"],
                                  [0.65, "#fee08b"], [1, "#d73027"]],
                       cmin=0, cmax=maximum if maximum > 0 else 1.0)
    xs, ys, zs, colors = [], [], [], []
    for element in model.elements:
        if element.i not in values or element.j not in values:
            continue
        for node_id in (element.i, element.j):
            node = model.nodes[node_id]
            xs.append(node.x)
            ys.append(node.y)
            zs.append(node.z)
            colors.append(values[node_id])
        xs.append(None)
        ys.append(None)
        zs.append(None)
        colors.append(0)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=colors, width=6, **color_range),
    ))
    fig.add_trace(go.Scatter3d(
        x=[model.nodes[n].x for n in node_ids],
        y=[model.nodes[n].y for n in node_ids],
        z=[model.nodes[n].z for n in node_ids],
        mode="markers", name="Перемещения узлов", showlegend=False,
        marker=dict(size=3, color=[values[n] for n in node_ids], **color_range,
                    showscale=True, colorbar=dict(title=f"{label}, мм",
                    **({"tickvals": [0], "ticktext": ["0"]} if maximum == 0 else {}))),
        customdata=[[n, *vectors[n], totals[n]] for n in node_ids],
        hovertemplate="Узел %{customdata[0]:.0f}<br>Ux = %{customdata[1]:.3f} мм"
                      "<br>Uy = %{customdata[2]:.3f} мм<br>Uz = %{customdata[3]:.3f} мм"
                      "<br>|U| = %{customdata[4]:.3f} мм<extra></extra>",
    ))
    fig.update_layout(title=dict(
        text=f"Перемещения {label} · максимум {maximum:.3f} мм · узел {maximum_node}",
        font=dict(size=15),
    ))


def _value_color(value: float, vmin: float, vmax: float, colorscale: str) -> str:
    normalized = (value - vmin) / (vmax - vmin)
    return sample_colorscale(colorscale, [normalized])[0]


def _add_roof(fig: go.Figure, data: CanopyInput, model: StructuralModel) -> None:
    plan = CanopyPlan(width=data.width, left_depth=data.left_depth, right_depth=data.right_depth)
    roof = roof_plane_from_input(data)
    points = model.roof_outline or [(point.x, point.y) for point in plan.points]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [roof.z(*point) for point in points]
    fig.add_trace(
        go.Mesh3d(
            x=xs,
            y=ys,
            z=zs,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#9ecae1",
            opacity=0.25,
            name="Кровля",
            hoverinfo="skip",
        )
    )


def _add_supports(fig: go.Figure, model: StructuralModel) -> None:
    xs, ys, zs, labels = [], [], [], []
    for column in model.columns:
        node = model.nodes[column.base_node]
        xs.append(node.x)
        ys.append(node.y)
        zs.append(node.z)
        labels.append(column.name)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers+text", text=labels,
        textposition="top center", name="Опоры", marker=dict(size=4, color="black"),
        hovertemplate="Стойка %{text}<extra></extra>",
    ))
