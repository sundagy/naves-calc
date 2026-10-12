from math import dist

import pytest

from src.config import input_from_preset
from src.fem.opensees_solver import solve_model
from src.geometry.naves_one import FRONT_COLUMN_NAMES, REAR_COLUMN_NAMES, generate_naves_one
from src.loads.self_weight import G_ACCEL, mass_breakdown
from src.sections.database import section_database
from src.visualization.model3d import model_figure


def test_crossbeams_support_both_rows_and_separate_tube_surfaces():
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    beams = [e for e in model.elements if e.role == "crossbeam"]
    assert len(beams) == 7
    assert {e.section for e in beams} == {"100x100x3"}
    for row, names in (("front", FRONT_COLUMN_NAMES), ("rear", REAR_COLUMN_NAMES)):
        segments = [e for e in beams if e.group == row]
        beam_nodes = [segments[0].i, *[e.j for e in segments]]
        for node_id, name in zip(beam_nodes, names, strict=True):
            beam = model.nodes[node_id]
            column = next(c for c in model.columns if c.name == name)
            top = model.nodes[column.top_node]
            assert (beam.x, beam.y) == (top.x, top.y)
            assert beam.z - 0.05 == pytest.approx(top.z)
            assert (node_id, top.id) in model.bearing_links
            truss = next(model.nodes[b] for a, b in model.bearing_links
                         if a == node_id and "lower_bottom" in model.nodes[b].roles)
            assert truss.z - section_database()[data.lower_chord_section].height / 2 == pytest.approx(beam.z + 0.05)
    mass = sum(dist(model.nodes[e.i].coords, model.nodes[e.j].coords)
               * section_database()[e.section].mass_per_m for e in beams)
    assert mass_breakdown(model)["crossbeams"] == pytest.approx(mass)
    geometry = model_figure(data, model)
    assert any("crossbeam" in trace.name for trace in geometry.data)
    realistic = model_figure(data, model, mode="REALISTIC")
    assert "Перекладина rear<br>100x100x3 мм" in realistic.data[0].hovertext


def test_crossbeam_bearings_transfer_load_and_mass_to_columns():
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    result = solve_model(data, model)
    assert result.ok
    lb = result.load_breakdown
    assert lb.crossbeams_self_weight_kN == pytest.approx(mass_breakdown(model)["crossbeams"] * G_ACCEL / 1000)
    assert lb.total_self_weight_kN == pytest.approx(
        lb.lower_trusses_self_weight_kN + lb.upper_trusses_self_weight_kN
        + lb.columns_self_weight_kN + lb.crossbeams_self_weight_kN)
    assert lb.sum_vertical_reactions_kN == pytest.approx(lb.total_applied_vertical_load_kN, abs=1e-6)
    displacement = result.nodes.set_index("node")["Uz_mm"]
    for retained, constrained in model.bearing_links:
        assert displacement[retained] == pytest.approx(displacement[constrained], abs=1e-9)
