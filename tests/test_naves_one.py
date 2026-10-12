from math import dist, hypot, isclose

import numpy as np
import pytest

from src.config import input_from_preset
from src.geometry.canopy import roof_plane_from_input
from src.fem.opensees_solver import solve_model
from src.geometry.naves_one import generate_naves_one, left_x
from src.loads.self_weight import mass_breakdown
from src.loads.tributary import tributary_roof_loads
from src.sections.database import section_database


def test_naves_one_drawing_geometry_and_load_path():
    data = input_from_preset("naves_one")
    # Compare the original drawing dimensions, not the edited preset.
    data.lower_web_section = "40x60x2"
    data.column_section = "100x100x3"
    data.column_height_mm = None
    data.lower_height_mm = 900
    data.upper_height_mm = 300
    data.upper_end_height_mm = None
    data.upper_max_diagonal_mm = None
    model = generate_naves_one(data)

    assert len(model.columns) == 12
    assert all(column.name != "K6" for column in model.columns)
    restored = next(column for column in model.columns if column.name == "K4-middle-2")
    assert restored.base_node in model.support_node_ids
    assert isclose(restored.x, 12.0) and isclose(restored.y, 4.829)
    assert not any(isclose(model.nodes[node_id].x, left_x(2.659)) and isclose(model.nodes[node_id].y, 2.659) for node_id in model.support_node_ids)
    assert not any(isclose(model.nodes[node_id].y, 2.659) for node_id in model.lower_trusses["B1"])
    assert len(model.lower_trusses) == 5
    assert len(model.upper_trusses) == 10
    offset = 0.1 + section_database()[data.lower_chord_section].height / 2
    assert isclose(model.nodes[model.columns[0].top_node].z, 2.601 - offset, abs_tol=0.002)
    assert isclose(model.nodes[model.columns[4].top_node].z, 2.400 - offset, abs_tol=0.002)
    assert isclose(sum(model.roof_tributary_areas.values()), model.loaded_roof_area, abs_tol=1e-8)
    assert isclose(sum(tributary_roof_loads(data, model, 1.0).values()), -model.loaded_roof_area * 1000, abs_tol=1e-6)

    # Compare with the drawing schedule minus K6 (23.42 kg).
    schedule_mass = (2882.91 - 23.42 + mass_breakdown(model)["crossbeams"]
                     - 9 * offset * section_database()[data.column_section].mass_per_m)
    assert abs(mass_breakdown(model)["total"] - schedule_mass) / schedule_mass < 0.05


def test_naves_one_trapezoids_and_shared_support_nodes():
    data = input_from_preset("naves_one")
    data.lower_height_mm = 500
    model = generate_naves_one(data)
    lower_top_ids = {node_id for row in model.lower_trusses.values() for node_id in row}
    lower_bottoms = {
        (node.x, node.y): node for node in model.nodes.values() if "lower_bottom" in node.roles
    }
    for node_id in lower_top_ids:
        top = model.nodes[node_id]
        # Upper-truss bearing nodes split only the top chord, not the web/bottom chord.
        bottom = lower_bottoms.get((top.x, top.y))
        if bottom is not None:
            assert isclose(top.z - bottom.z, 0.5, abs_tol=1e-8)

    for name, top_ids in model.upper_trusses.items():
        tops = [model.nodes[node_id] for node_id in top_ids]
        bottom_ids = model.upper_bottom_trusses[name]
        bottoms = [model.nodes[node_id] for node_id in bottom_ids]
        assert isclose(tops[0].z - bottoms[0].z, 0.4, abs_tol=1e-8)
        end_height = tops[-1].z - bottoms[-1].z
        if name == "F1":
            assert isclose(end_height, 0.17, abs_tol=1e-8)
        else:
            assert 0.17 < end_height < 0.4
        if name == "F9":
            assert isclose(end_height, 0.260938, abs_tol=1e-6)
        if name == "F10":
            assert isclose(end_height, 0.319918, abs_tol=1e-6)
        for top, bottom in zip(tops, bottoms):
            assert isclose(top.x, bottom.x, abs_tol=1e-8)
            assert isclose(top.y, bottom.y, abs_tol=1e-8)
            t = (top.x - tops[0].x) / (tops[-1].x - tops[0].x)
            assert isclose(bottom.z, bottoms[0].z + t * (bottoms[-1].z - bottoms[0].z), abs_tol=1e-8)
        # Bearings tie translations while keeping upper/lower rotations separate.
        for node_id in lower_top_ids:
            lower_top = model.nodes[node_id]
            if isclose(lower_top.y, tops[0].y, abs_tol=1e-8):
                bearing = dict(model.bearing_links)[node_id]
                assert bearing in bottom_ids
                assert model.nodes[bearing].coords == lower_top.coords

    # Each complete chord surface must fit one plane, not just each F truss.
    for rows in (model.upper_trusses, model.upper_bottom_trusses):
        nodes = [model.nodes[node_id] for row in rows.values() for node_id in row]
        xy1 = np.array([(node.x, node.y, 1.0) for node in nodes])
        z = np.array([node.z for node in nodes])
        plane, *_ = np.linalg.lstsq(xy1, z, rcond=None)
        assert np.max(np.abs(xy1 @ plane - z)) < 1e-8


@pytest.mark.parametrize("lower,start,end", [(500, 400, 170), (600, 450, 200), (500, 300, 300)])
def test_naves_one_bearing_elevations_with_editable_truss_heights(lower, start, end):
    data = input_from_preset("naves_one")
    data.lower_height_mm = lower
    data.upper_height_mm = start
    data.upper_end_height_mm = end
    model = generate_naves_one(data)
    roof = roof_plane_from_input(data)
    for column in model.columns:
        base = model.nodes[column.base_node]
        top = model.nodes[column.top_node]
        assert base.z == 0.0
        offset = (0.1 + section_database()[data.lower_chord_section].height / 2
                  if column.name == "K7" or column.name.endswith(("-front", "-rear")) else 0.0)
        assert isclose(top.z - base.z, 2.7 - offset, abs_tol=1e-8)
    for row in model.upper_bottom_trusses.values():
        for node_id in row:
            assert isclose(model.nodes[node_id].z, 2.7 + lower / 1000, abs_tol=1e-8)
    for node_id in model.roof_node_ids:
        node = model.nodes[node_id]
        assert isclose(node.z, roof.z(node.x, node.y), abs_tol=1e-8)
    for index, expected in ((0, start), (-1, end)):
        top = model.nodes[model.upper_trusses["F1"][index]]
        bottom = model.nodes[model.upper_bottom_trusses["F1"][index]]
        assert isclose(top.z - bottom.z, expected / 1000, abs_tol=1e-8)


def test_naves_one_vertical_reactions_balance():
    data = input_from_preset("naves_one")
    result = solve_model(data, generate_naves_one(data))
    assert result.ok, result.message
    assert abs(result.load_breakdown.total_applied_vertical_load_kN - result.load_breakdown.sum_vertical_reactions_kN) < 1e-6


@pytest.mark.parametrize("web_type", ["WARREN", "PRATT", "HOWE", "CROSS"])
@pytest.mark.parametrize("start,end", [(400, 170), (450, 200), (170, 400)])
def test_upper_diagonals_do_not_exceed_500_mm(web_type, start, end):
    data = input_from_preset("naves_one")
    data.upper_max_diagonal_mm = 500
    data.upper_web_type = web_type
    data.upper_height_mm, data.upper_end_height_mm = start, end
    model = generate_naves_one(data)
    checked_groups = set()
    for element in model.elements:
        if element.role != "upper_web":
            continue
        a, b = model.nodes[element.i], model.nodes[element.j]
        if hypot(a.x - b.x, a.y - b.y) < 1e-7:
            continue
        assert 0 < dist(a.coords, b.coords) <= 0.5
        checked_groups.add(element.group)
    assert checked_groups == set(model.upper_trusses)


def test_upper_height_incompatible_with_diagonal_limit():
    data = input_from_preset("naves_one")
    data.upper_max_diagonal_mm = 500
    data.upper_height_mm = 500
    with pytest.raises(ValueError, match="высоты верхних ферм должны быть меньше"):
        generate_naves_one(data)
