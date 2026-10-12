from math import ceil, dist

import pytest

from src.config import input_from_preset
from src.fem.opensees_solver import solve_model
from src.geometry import naves_one


def web_geometry(model):
    return sorted((e.group, model.nodes[e.i].coords, model.nodes[e.j].coords)
                  for e in model.elements if e.role == "lower_web")


@pytest.mark.parametrize("web_type", ["WARREN", "PRATT", "HOWE", "CROSS"])
def test_lower_web_does_not_change_when_upper_trusses_move(monkeypatch, web_type):
    data = input_from_preset("naves_one")
    data.lower_web_type = web_type
    original = naves_one.generate_naves_one(data)
    monkeypatch.setattr(naves_one, "UPPER_YS", (0.0, 0.8, 2.2, 3.4, 4.3, 5.2, 6.4, 7.3, 8.0, 8.696))
    moved = naves_one.generate_naves_one(data)
    assert web_geometry(moved) == web_geometry(original)
    # Every F/B intersection remains a shared node, including off-panel bearings.
    lower_tops = {node for row in moved.lower_trusses.values() for node in row}
    web_nodes = {node for e in moved.elements if e.role == "lower_web" for node in (e.i, e.j)}
    off_panel = set()
    for row in moved.upper_bottom_trusses.values():
        y = moved.nodes[row[0]].y
        intersections = {node for node in lower_tops if moved.nodes[node].y == y}
        assert intersections <= set(row)
        off_panel.update(intersections - web_nodes)
    assert off_panel
    for node in off_panel:
        chords = [e for e in moved.elements if e.role == "lower_chord" and node in (e.i, e.j)]
        assert len(chords) == 2
        assert all(e.is_beam for e in chords)
    result = solve_model(data, moved)
    assert result.ok
    assert result.load_breakdown.sum_vertical_reactions_kN == pytest.approx(
        result.load_breakdown.total_applied_vertical_load_kN, abs=1e-6)


def test_lower_web_responds_to_its_own_panel_length():
    data = input_from_preset("naves_one")
    coarse = naves_one.generate_naves_one(data)
    data.lower_target_panel_length = 0.5
    fine = naves_one.generate_naves_one(data)
    assert len(web_geometry(fine)) > len(web_geometry(coarse))
    assert [coarse.nodes[n].coords for n in coarse.roof_node_ids] == [
        fine.nodes[n].coords for n in fine.roof_node_ids]


@pytest.mark.parametrize("step", [0.5, 1.0, 1.5])
def test_lower_web_is_uniform_and_columns_only_split_bottom_chord(step):
    data = input_from_preset("naves_one")
    data.lower_target_panel_length = step
    model = naves_one.generate_naves_one(data)
    web_nodes = {node for e in model.elements if e.role == "lower_web" for node in (e.i, e.j)}
    for group, row in model.lower_trusses.items():
        first, last = model.nodes[row[0]], model.nodes[row[-1]]
        length = dist((first.x, first.y), (last.x, last.y))
        count = ceil(length / step)
        stations = sorted({model.nodes[n].y for e in model.elements
                           if e.role == "lower_web" and e.group == group for n in (e.i, e.j)})
        assert stations == [round(last.y * i / count, 8) for i in range(count + 1)]
    for column in model.columns:
        if column.name not in ("K5", "K4-middle-1", "K4-middle-2"):
            continue
        assert column.top_node not in web_nodes
        chords = [e for e in model.elements if e.role == "lower_chord"
                  and column.top_node in (e.i, e.j)]
        assert len(chords) == 2
        assert all(e.is_beam and "bottom chord" in e.label for e in chords)
    result = solve_model(data, model)
    assert result.ok
    assert result.load_breakdown.sum_vertical_reactions_kN == pytest.approx(
        result.load_breakdown.total_applied_vertical_load_kN, abs=1e-6)
