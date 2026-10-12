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
            assert beam.y > top.y if row == "front" else beam.y < top.y
            assert top.z - (beam.z - 0.05) == pytest.approx(data.lower_height_mm / 1000 + 0.1)
            column_elements = [e for e in model.elements if e.role == "column" and e.group == name]
            seat = next(model.nodes[b] for a, b in model.bearing_links
                        if a == node_id and "column_joint" in model.nodes[b].roles)
            assert seat.coords == pytest.approx((column.x, column.y, beam.z))
            assert any(e.j == seat.id for e in column_elements)
            assert any(e.i == seat.id for e in column_elements)
            assert all((model.nodes[n].x, model.nodes[n].y) == (column.x, column.y)
                       for e in column_elements for n in (e.i, e.j))
            upper = next(model.nodes[a] for a, b in model.bearing_links if b == top.id)
            assert "lower_top" in upper.roles
            assert upper.z - section_database()[data.lower_chord_section].height / 2 == pytest.approx(top.z)
            truss = next(model.nodes[b] for a, b in model.bearing_links
                         if a == node_id and "lower_bottom" in model.nodes[b].roles)
            assert (beam.x, beam.y) == (truss.x, truss.y)
            assert truss.z - section_database()[data.lower_chord_section].height / 2 == pytest.approx(beam.z + 0.05)
    mass = sum(dist(model.nodes[e.i].coords, model.nodes[e.j].coords)
               * section_database()[e.section].mass_per_m for e in beams)
    assert mass_breakdown(model)["crossbeams"] == pytest.approx(mass)
    geometry = model_figure(data, model)
    assert any("crossbeam" in trace.name for trace in geometry.data)
    realistic = model_figure(data, model, mode="REALISTIC")
    assert "Перекладина rear<br>100x100x3 мм" in realistic.data[0].hovertext


@pytest.mark.parametrize("spatial", [False, True])
def test_crossbeam_bearings_transfer_load_and_mass_to_columns(spatial):
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    data.spatial_frame = spatial
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
        expected = displacement[retained]
        if spatial:
            import openseespy.opensees as ops
            a, b = model.nodes[retained], model.nodes[constrained]
            rx, ry, _ = ops.nodeDisp(retained)[3:]
            expected += (rx * (b.y - a.y) - ry * (b.x - a.x)) * 1000
        assert expected == pytest.approx(displacement[constrained], abs=1e-9)


@pytest.mark.parametrize("column_section", ["100x100x3", "120x120x3"])
def test_every_b_end_overlaps_the_crossbeam_top_surface(column_section):
    data = input_from_preset("naves_one")
    data.column_section = column_section
    model = generate_naves_one(data)
    sections = section_database()
    for group, row in model.lower_trusses.items():
        for end, beam_row in ((row[0], "front"), (row[-1], "rear")):
            upper = model.nodes[end]
            bottom = next(n for n in model.nodes.values() if "lower_bottom" in n.roles
                          and (n.x, n.y) == (upper.x, upper.y))
            contacts = []
            for beam in model.elements:
                if beam.role != "crossbeam" or beam.group != beam_row:
                    continue
                a, b = model.nodes[beam.i], model.nodes[beam.j]
                dx, dy = b.x - a.x, b.y - a.y
                t = ((bottom.x - a.x) * dx + (bottom.y - a.y) * dy) / (dx * dx + dy * dy)
                if not -1e-8 <= t <= 1 + 1e-8:
                    continue
                distance = dist((bottom.x, bottom.y), (a.x + t * dx, a.y + t * dy))
                if distance <= sections[beam.section].width / 2:
                    contacts.append(a.z + t * (b.z - a.z) + sections[beam.section].height / 2)
            assert contacts, (group, beam_row)
            underside = bottom.z - sections[data.lower_chord_section].height / 2
            assert any(abs(underside - z) < 1e-8 for z in contacts), (group, beam_row)
