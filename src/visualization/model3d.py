from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from src.geometry.canopy import CanopyPlan, roof_plane_from_input
from src.models import CanopyInput, StructuralModel


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
    _add_roof(fig, data)

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

    if mode == "DEFORMED" and disp:
        _add_elements(fig, model, {}, "Исходная", opacity=0.25)
        _add_elements(fig, model, disp, "Деформированная", deformation_scale, opacity=1.0)
    elif element_values:
        _add_colored_elements(fig, model, element_values, mode)
    else:
        _add_elements(fig, model, {}, "Геометрия", opacity=1.0)

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
    return fig


def _node_xyz(model: StructuralModel, node_id: int, disp: dict, scale: float) -> tuple[float, float, float]:
    node = model.nodes[node_id]
    dx, dy, dz = disp.get(node_id, (0.0, 0.0, 0.0))
    return node.x + dx * scale, node.y + dy * scale, node.z + dz * scale


def _add_elements(fig: go.Figure, model: StructuralModel, disp: dict, name: str, scale: float = 1.0, opacity: float = 1.0) -> None:
    for role, color in ROLE_COLORS.items():
        xs, ys, zs = [], [], []
        for element in model.elements:
            if element.role != role:
                continue
            p1 = _node_xyz(model, element.i, disp, scale)
            p2 = _node_xyz(model, element.j, disp, scale)
            xs += [p1[0], p2[0], None]
            ys += [p1[1], p2[1], None]
            zs += [p1[2], p2[2], None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=f"{name}: {role}", line=dict(color=color, width=5), opacity=opacity))


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
                hovertext=f"E{element.id} {element.role}<br>{value:.3f}",
            )
        )


def _value_color(value: float, vmin: float, vmax: float, colorscale: str) -> str:
    normalized = (value - vmin) / (vmax - vmin)
    return sample_colorscale(colorscale, [normalized])[0]


def _add_roof(fig: go.Figure, data: CanopyInput) -> None:
    plan = CanopyPlan(width=data.width, left_depth=data.left_depth, right_depth=data.right_depth)
    roof = roof_plane_from_input(data)
    points = plan.points
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [roof.z(point.x, point.y) for point in points]
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
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers+text", text=labels, textposition="top center", name="Опоры", marker=dict(size=4, color="black")))
