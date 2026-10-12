import numpy as np
import pytest

from src.config import input_from_preset
from src.geometry.naves_one import generate_naves_one
from src.sections.rhs import parse_rhs
from src.visualization.model3d import _rhs_mesh, model_figure


@pytest.mark.parametrize("end,width_axis", [
    ((0, 0, 2.7), (1, 0, 0)),
    ((3, 0, 0), (0, 1, 0)),
    ((0, 2, 1), (1, 0, 0)),
])
def test_hollow_profile_dimensions_and_volume(end, width_axis):
    section = parse_rhs("60x40x3")
    points, faces = _rhs_mesh((0, 0, 0), end, section, width_axis)
    assert np.isfinite(points).all()
    assert np.linalg.norm(points[0] - points[1]) == pytest.approx(0.06)
    assert np.linalg.norm(points[1] - points[2]) == pytest.approx(0.04)
    assert np.linalg.norm(points[8] - points[9]) == pytest.approx(0.054)
    assert np.linalg.norm(points[9] - points[10]) == pytest.approx(0.034)
    # Signed surface volume checks winding and hollow walls, not a solid box.
    a, b, c = (points[faces[:, i]] for i in range(3))
    volume = np.sum(a * np.cross(b, c)) / 6
    assert volume == pytest.approx(section.area * np.linalg.norm(end))


def test_realistic_view_includes_every_member_and_hover_labels():
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    figure = model_figure(data, model, mode="REALISTIC")
    assert len(figure.data) == 1
    mesh = figure.data[0]
    assert mesh.type == "mesh3d"
    assert len(mesh.x) == 16 * len(model.elements)
    assert len(mesh.i) == 32 * len(model.elements)
    assert np.isfinite(np.array([mesh.x, mesh.y, mesh.z])).all()
    assert "Стойка K5<br>120x120x3 мм" in mesh.hovertext
    assert "Ферма F10<br>60x60x3 мм" in mesh.hovertext
    assert mesh.flatshading


@pytest.mark.parametrize("column_section", ["100x100x3", "120x120x3"])
def test_b_end_vertical_surfaces_touch_column_faces(column_section):
    data = input_from_preset("naves_one")
    data.column_section = column_section
    model = generate_naves_one(data)
    column_depth = parse_rhs(column_section).height
    for group, row in model.lower_trusses.items():
        first, last = model.nodes[row[0]], model.nodes[row[-1]]
        width_axis = (-(last.y - first.y), last.x - first.x, 0.0)
        for endpoint, front in ((first, True), (last, False)):
            vertical = next(e for e in model.elements
                            if e.group == group and e.role == "lower_web"
                            and endpoint.id in (e.i, e.j)
                            and model.nodes[e.i].y == model.nodes[e.j].y)
            bottom = vertical.j if vertical.i == endpoint.id else vertical.i
            assert any(b == bottom and "crossbeam" in model.nodes[a].roles
                       for a, b in model.bearing_links)
            points, _ = _rhs_mesh(model.nodes[vertical.i].coords, model.nodes[vertical.j].coords,
                                  parse_rhs(vertical.section), width_axis)
            column = min(model.columns, key=lambda c: np.hypot(c.x - endpoint.x, c.y - endpoint.y))
            face_y = column.y + (column_depth / 2 if front else -column_depth / 2)
            outer_y = points[:, 1].min() if front else points[:, 1].max()
            assert outer_y == pytest.approx(face_y, abs=1e-8)
        assert all(model.nodes[a].y < model.nodes[b].y for a, b in zip(row, row[1:]))
