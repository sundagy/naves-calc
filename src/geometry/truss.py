from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import ceil, isclose

from src.geometry.canopy import CanopyPlan, lower_truss_positions, roof_plane_from_input, upper_truss_positions
from src.geometry.columns import build_column_points, validate_columns
from src.models import CanopyInput, Column, Element, Node, StructuralModel, WebType


COORD_ROUND = 8


@dataclass
class NodeRegistry:
    nodes: dict[int, Node] = field(default_factory=dict)
    by_coord: dict[tuple[float, float, float], int] = field(default_factory=dict)
    next_id: int = 1

    def get(self, x: float, y: float, z: float, label: str, role: str) -> int:
        key = (round(x, COORD_ROUND), round(y, COORD_ROUND), round(z, COORD_ROUND))
        if key in self.by_coord:
            node = self.nodes[self.by_coord[key]]
            node.roles.add(role)
            return node.id
        node_id = self.next_id
        self.next_id += 1
        self.by_coord[key] = node_id
        self.nodes[node_id] = Node(node_id, float(x), float(y), float(z), label, {role})
        return node_id


@dataclass
class ElementRegistry:
    elements: list[Element] = field(default_factory=list)
    next_id: int = 1
    existing_pairs: set[tuple[int, int, str]] = field(default_factory=set)

    def add(self, i: int, j: int, role: str, group: str, section: str, label: str) -> None:
        if i == j:
            return
        pair = (min(i, j), max(i, j), role)
        if pair in self.existing_pairs:
            return
        self.existing_pairs.add(pair)
        self.elements.append(Element(self.next_id, i, j, role, group, section, label))
        self.next_id += 1


def generate_model(data: CanopyInput) -> StructuralModel:
    plan = CanopyPlan(width=data.width, left_depth=data.left_depth, right_depth=data.right_depth)
    roof = roof_plane_from_input(data)
    lower_xs = lower_truss_positions(data)
    upper_ts = upper_truss_positions(data)
    column_points = build_column_points(data, plan)
    validate_columns(data, plan, column_points)

    lower_height = data.lower_height_mm / 1000.0
    upper_height = data.upper_height_mm / 1000.0
    nodes = NodeRegistry()
    elements = ElementRegistry()
    lower_top: dict[tuple[int, float], int] = {}
    lower_bottom: dict[tuple[int, float], int] = {}
    roof_nodes: dict[tuple[int, int], int] = {}
    lower_trusses: dict[str, list[int]] = {}
    upper_trusses: dict[str, list[int]] = {}
    upper_bottom_trusses: dict[str, list[int]] = {}

    column_stations: dict[int, list[float]] = defaultdict(list)
    for x_index, x in enumerate(lower_xs):
        depth = plan.depth(x)
        for _, (column_x, column_y) in column_points.items():
            if isclose(column_x, x, abs_tol=1e-7) and -1e-7 <= column_y <= depth + 1e-7:
                column_stations[x_index].append(column_y / depth if depth else 0.0)

    for x_index, x in enumerate(lower_xs):
        station_ts = lower_panel_stations(data, plan.depth(x), upper_ts, column_stations[x_index])
        top_nodes: list[int] = []
        bottom_nodes: list[int] = []
        truss_name = f"F{x_index + 1}"
        for station_t in station_ts:
            y = station_t * plan.depth(x)
            top_z = roof.z(x, y) - upper_height
            bottom_z = top_z - lower_height
            top_id = nodes.get(x, y, top_z, f"{truss_name}-LT-{station_t:.3f}", "lower_top")
            bottom_id = nodes.get(x, y, bottom_z, f"{truss_name}-LB-{station_t:.3f}", "lower_bottom")
            lower_top[(x_index, round(station_t, COORD_ROUND))] = top_id
            lower_bottom[(x_index, round(station_t, COORD_ROUND))] = bottom_id
            top_nodes.append(top_id)
            bottom_nodes.append(bottom_id)
        for a, b in zip(top_nodes, top_nodes[1:]):
            elements.add(a, b, "lower_chord", truss_name, data.lower_chord_section, f"{truss_name} top chord")
        for a, b in zip(bottom_nodes, bottom_nodes[1:]):
            elements.add(a, b, "lower_chord", truss_name, data.lower_chord_section, f"{truss_name} bottom chord")
        add_web_pattern(elements, top_nodes, bottom_nodes, "lower_web", truss_name, data.lower_web_section, WebType(data.lower_web_type))
        lower_trusses[truss_name] = top_nodes

    for t_index, station_t in enumerate(upper_ts):
        upper_name = f"U{t_index + 1}"
        top_nodes = []
        bottom_nodes = []
        for x_panel_index, x in enumerate(upper_panel_positions(data, lower_xs)):
            y = station_t * plan.depth(x)
            roof_z = roof.z(x, y)
            top_id = nodes.get(x, y, roof_z, f"{upper_name}-UT-{x_panel_index + 1}", "roof")
            lower_x_index = matching_lower_x_index(x, lower_xs)
            if lower_x_index is None:
                bottom_z = roof_z - upper_height
                bottom_id = nodes.get(x, y, bottom_z, f"{upper_name}-UB-{x_panel_index + 1}", "upper_bottom")
            else:
                bottom_id = lower_top[(lower_x_index, round(station_t, COORD_ROUND))]
            top_nodes.append(top_id)
            bottom_nodes.append(bottom_id)
            roof_nodes[(x_panel_index, t_index)] = top_id
        for a, b in zip(top_nodes, top_nodes[1:]):
            elements.add(a, b, "upper_chord", upper_name, data.upper_chord_section, f"{upper_name} top chord")
        for a, b in zip(bottom_nodes, bottom_nodes[1:]):
            elements.add(a, b, "upper_chord", upper_name, data.upper_chord_section, f"{upper_name} bottom chord")
        add_web_pattern(elements, top_nodes, bottom_nodes, "upper_web", upper_name, data.upper_web_section, WebType(data.upper_web_type))
        upper_trusses[upper_name] = top_nodes
        upper_bottom_trusses[upper_name] = bottom_nodes

    columns: list[Column] = []
    support_node_ids: list[int] = []
    for name, (x, y) in column_points.items():
        top_z = roof.z(x, y) - upper_height
        top_id = nodes.get(x, y, top_z, f"{name}-top", "column_top")
        base_id = nodes.get(x, y, 0.0, f"{name}-base", "support")
        elements.add(base_id, top_id, "column", name, data.column_section, name)
        columns.append(Column(name, x, y, base_id, top_id))
        support_node_ids.append(base_id)

    model = StructuralModel(
        nodes=nodes.nodes,
        elements=elements.elements,
        columns=columns,
        lower_trusses=lower_trusses,
        upper_trusses=upper_trusses,
        upper_bottom_trusses=upper_bottom_trusses,
        roof_node_ids=list(roof_nodes.values()),
        support_node_ids=support_node_ids,
        loaded_roof_area=plan.area,
    )
    validate_connectivity(model)
    return model


def validate_connectivity(model: StructuralModel) -> None:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for element in model.elements:
        adjacency[element.i].add(element.j)
        adjacency[element.j].add(element.i)

    fixed = set(model.support_node_ids)
    for column in model.columns:
        if column.top_node not in adjacency or column.base_node not in adjacency[column.top_node]:
            raise ValueError(f"Column {column.name} is not connected to the structural graph.")
        if "lower_top" not in model.nodes[column.top_node].roles:
            raise ValueError(f"Column {column.name} top node is not a real lower truss node.")

    for roof_node_id in model.roof_node_ids:
        visited = {roof_node_id}
        queue = deque([roof_node_id])
        reaches_support = False
        while queue:
            node_id = queue.popleft()
            if node_id in fixed:
                reaches_support = True
                break
            for next_id in adjacency[node_id]:
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append(next_id)
        if not reaches_support:
            raise ValueError(f"Roof node {roof_node_id} has no load path to support.")


def upper_panel_positions(data: CanopyInput, lower_xs: list[float]) -> list[float]:
    if data.panel_count is not None:
        panel_count = max(1, data.panel_count)
    else:
        panel_count = max(1, ceil(data.width / max(data.target_panel_length, 0.1)))
    uniform = [data.width * index / panel_count for index in range(panel_count + 1)]
    return sorted({round(x, COORD_ROUND) for x in [*uniform, *lower_xs]})


def lower_panel_stations(
    data: CanopyInput,
    truss_depth: float,
    upper_ts: list[float],
    support_ts: list[float],
) -> list[float]:
    panel_count = max(1, ceil(truss_depth / max(data.lower_target_panel_length, 0.1)))
    panel_ts = [index / panel_count for index in range(panel_count + 1)]
    return sorted({round(t, COORD_ROUND) for t in [0.0, 1.0, *upper_ts, *support_ts, *panel_ts]})


def matching_lower_x_index(x: float, lower_xs: list[float]) -> int | None:
    for index, lower_x in enumerate(lower_xs):
        if isclose(x, lower_x, abs_tol=1e-7):
            return index
    return None


def add_web_pattern(
    elements: ElementRegistry,
    top_nodes: list[int],
    bottom_nodes: list[int],
    role: str,
    group: str,
    section: str,
    web_type: WebType,
) -> None:
    add_verticals(elements, top_nodes, bottom_nodes, role, group, section)
    for index in range(len(top_nodes) - 1):
        for i, j, label in panel_diagonals(top_nodes, bottom_nodes, index, web_type, group):
            elements.add(i, j, role, group, section, label)


def add_verticals(
    elements: ElementRegistry,
    top_nodes: list[int],
    bottom_nodes: list[int],
    role: str,
    group: str,
    section: str,
) -> None:
    for index, (top_node, bottom_node) in enumerate(zip(top_nodes, bottom_nodes), start=1):
        elements.add(top_node, bottom_node, role, group, section, f"{group} vertical {index}")


def panel_diagonals(
    top_nodes: list[int],
    bottom_nodes: list[int],
    index: int,
    web_type: WebType,
    group: str,
) -> list[tuple[int, int, str]]:
    descending = (top_nodes[index], bottom_nodes[index + 1], f"{group} {web_type.value} diagonal")
    ascending = (bottom_nodes[index], top_nodes[index + 1], f"{group} {web_type.value} diagonal")
    if web_type == WebType.WARREN:
        return [descending if index % 2 == 0 else ascending]
    if web_type == WebType.PRATT:
        center = (len(top_nodes) - 1) / 2.0
        return [descending if index < center else ascending]
    if web_type == WebType.HOWE:
        center = (len(top_nodes) - 1) / 2.0
        return [ascending if index < center else descending]
    if web_type == WebType.CROSS:
        return [descending, ascending]
    raise ValueError(f"Unsupported web type: {web_type}")
