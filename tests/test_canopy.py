from collections import defaultdict, deque

from src.config import input_from_preset
from src.geometry.canopy import CanopyPlan
from src.geometry.columns import EXISTING_10_COLUMNS, ORIGINAL_SUPPORTS_BY_TRUSS
from src.geometry.truss import generate_model
from src.models import SupportMode, WebType


def test_columns():
    data = input_from_preset("original")
    model = generate_model(data)
    assert len(model.columns) == 10
    actual = {column.name: (column.x, column.y) for column in model.columns}
    assert actual == EXISTING_10_COLUMNS


def test_original_can_be_parametric_with_lower_truss_end_supports():
    data = input_from_preset("original")
    data.lower_count = 6
    data.lower_positions_x = None
    data.support_mode = SupportMode.LOWER_TRUSS_ENDS
    model = generate_model(data)
    assert len(model.lower_trusses) == 6
    assert len(model.columns) == 14
    column_names = {column.name for column in model.columns}
    assert {"C9", "C10"} <= column_names
    assert all("lower_top" in model.nodes[column.top_node].roles for column in model.columns)


def test_side_columns_c9_c10_remain_in_parametric_support_mode():
    data = input_from_preset("original")
    data.lower_count = 8
    data.lower_positions_x = None
    data.support_mode = SupportMode.LOWER_TRUSS_ENDS
    model = generate_model(data)
    columns = {column.name: column for column in model.columns}
    assert (columns["C9"].x, columns["C9"].y) == (0.0, 3.5)
    assert (columns["C10"].x, columns["C10"].y) == (12.0, 3.5)
    assert "lower_top" in model.nodes[columns["C9"].top_node].roles
    assert "lower_top" in model.nodes[columns["C10"].top_node].roles


def test_lower_target_panel_length_controls_lower_truss_density():
    coarse_data = input_from_preset("original")
    coarse_data.lower_target_panel_length = 2.0
    coarse_model = generate_model(coarse_data)
    dense_data = input_from_preset("original")
    dense_data.lower_target_panel_length = 0.5
    dense_model = generate_model(dense_data)
    assert len(dense_model.lower_trusses["F1"]) > len(coarse_model.lower_trusses["F1"])
    assert 3.5 in [round(dense_model.nodes[node_id].y, 2) for node_id in dense_model.lower_trusses["F1"]]


def test_lower_truss_supports():
    data = input_from_preset("original")
    model = generate_model(data)
    by_coord = {(column.x, column.y): column.name for column in model.columns}
    supports_by_truss = {}
    for truss, node_ids in model.lower_trusses.items():
        names = []
        for node_id in node_ids:
            node = model.nodes[node_id]
            if (node.x, node.y) in by_coord:
                names.append(by_coord[(node.x, node.y)])
        supports_by_truss[truss] = names
    assert supports_by_truss == ORIGINAL_SUPPORTS_BY_TRUSS


def test_no_interior_columns():
    plan = CanopyPlan()
    data = input_from_preset("original")
    model = generate_model(data)
    for column in model.columns:
        assert not plan.contains_strictly(column.x, column.y)


def test_fem_connectivity_shared_support_nodes():
    model = generate_model(input_from_preset("original"))
    coords = {}
    for node in model.nodes.values():
        assert node.coords not in coords
        coords[node.coords] = node.id
    for column in model.columns:
        assert "lower_top" in model.nodes[column.top_node].roles


def test_upper_trusses_have_independent_dense_panels():
    model = generate_model(input_from_preset("original"))
    for top_nodes in model.upper_trusses.values():
        assert len(top_nodes) == 13
    upper_web_count = sum(1 for element in model.elements if element.role == "upper_web")
    assert upper_web_count == 11 * (12 + 13)


def test_upper_trusses_use_zigzag_not_cross_bracing():
    model = generate_model(input_from_preset("original"))
    for upper_name, top_nodes in model.upper_trusses.items():
        bottom_nodes = model.upper_bottom_trusses[upper_name]
        web_pairs = {
            frozenset((element.i, element.j))
            for element in model.elements
            if element.role == "upper_web" and element.group == upper_name
        }
        for index in range(len(top_nodes) - 1):
            diagonal_a = frozenset((top_nodes[index], bottom_nodes[index + 1]))
            diagonal_b = frozenset((bottom_nodes[index], top_nodes[index + 1]))
            assert (diagonal_a in web_pairs) != (diagonal_b in web_pairs)


def test_upper_lower_intersections_share_nodes():
    model = generate_model(input_from_preset("original"))
    for upper_name, bottom_nodes in model.upper_bottom_trusses.items():
        for node_id in bottom_nodes:
            node = model.nodes[node_id]
            if node.x in {0.0, 4.0, 8.0, 12.0}:
                assert "lower_top" in node.roles, upper_name


def test_supported_web_patterns_generate_models():
    for web_type in WebType:
        data = input_from_preset("original")
        data.lower_web_type = web_type
        data.upper_web_type = web_type
        model = generate_model(data)
        assert any(element.role == "lower_web" for element in model.elements)
        assert any(element.role == "upper_web" for element in model.elements)


def test_load_path_from_roof_to_supports():
    model = generate_model(input_from_preset("original"))
    adjacency = defaultdict(set)
    for element in model.elements:
        adjacency[element.i].add(element.j)
        adjacency[element.j].add(element.i)
    supports = set(model.support_node_ids)
    for roof_node in model.roof_node_ids:
        visited = {roof_node}
        queue = deque([roof_node])
        found = False
        while queue:
            current = queue.popleft()
            if current in supports:
                found = True
                break
            for nxt in adjacency[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        assert found
