from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from src.geometry.canopy import CanopyPlan, roof_plane_from_input
from src.models import CanopyInput, StructuralModel
from src.sections.database import section_database
from src.sections.rhs import RHSSection


ROLE_COLORS = {
    "column": "#777777",
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
) -> go.Figure:
    fig = go.Figure()
    if mode != "REALISTIC":
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

    if mode == "REALISTIC":
        _add_realistic_elements(fig, model)
    elif mode == "DEFORMED" and disp:
        _add_elements(fig, model, {}, "Исходная", opacity=0.25)
        _add_elements(fig, model, disp, "Деформированная", deformation_scale, opacity=1.0)
    elif element_values:
        _add_colored_elements(fig, model, element_values, mode)
    else:
        _add_elements(fig, model, {}, "Геометрия", opacity=1.0)

    if mode != "REALISTIC":
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
    if mode == "REALISTIC":
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                bgcolor="#f2f4f6",
            ),
            paper_bgcolor="#f2f4f6", showlegend=False,
        )
    return fig


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


def _add_realistic_elements(fig: go.Figure, model: StructuralModel) -> None:
    sections = section_database()
    vertices, triangles, labels = [], [], []
    for element in model.elements:
        if element.role == "column":
            width_axis = (1.0, 0.0, 0.0)
        else:
            trusses = model.lower_trusses if element.role.startswith("lower") else model.upper_trusses
            row = trusses[element.group]
            first, last = model.nodes[row[0]], model.nodes[row[-1]]
            width_axis = (-(last.y - first.y), last.x - first.x, 0.0)
        points, faces = _rhs_mesh(
            model.nodes[element.i].coords, model.nodes[element.j].coords,
            sections[element.section], width_axis,
        )
        triangles.extend(faces + len(vertices))
        vertices.extend(points)
        label = f"{'Стойка' if element.role == 'column' else 'Ферма'} {element.group}<br>{element.section} мм"
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
        name="Стальной каркас", showlegend=False,
    ))


def _node_xyz(model: StructuralModel, node_id: int, disp: dict, scale: float) -> tuple[float, float, float]:
    node = model.nodes[node_id]
    dx, dy, dz = disp.get(node_id, (0.0, 0.0, 0.0))
    return node.x + dx * scale, node.y + dy * scale, node.z + dz * scale


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
            label = f"{'Стойка' if role == 'column' else 'Ферма'} {element.group}"
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
                    f"{'Стойка' if element.role == 'column' else 'Ферма'} {element.group}<br>"
                    + (f"N = {value:.3f} кН" if mode == "AXIAL_FORCE" else f"Util = {value:.3f}")
                ),
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )


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
