from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import input_from_preset
from src.geometry.naves_one import generate_naves_one
from src.visualization.model3d import model_figure


@pytest.mark.parametrize("component,maximum", [("UZ", 12), ("TOTAL", 13)])
def test_displacement_map_units_components_and_original_geometry(component, maximum):
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    rows = [{"node": n, "Ux_mm": 3, "Uy_mm": 4, "Uz_mm": -12} for n in model.nodes]
    result = SimpleNamespace(ok=True, nodes=pd.DataFrame(rows), elements=pd.DataFrame())
    fig = model_figure(data, model, result, mode="DISPLACEMENT", deformation_scale=100,
                       displacement_component=component)
    points = next(t for t in fig.data if t.name == "Перемещения узлов")
    assert list(points.marker.color) == pytest.approx([maximum] * len(rows))
    assert points.marker.cmin == 0
    assert points.marker.cmax == maximum
    assert list(points.customdata[0][1:]) == pytest.approx([3, 4, -12, 13])
    assert list(zip(points.x, points.y, points.z)) == [n.coords for n in model.nodes.values()]
    lines = next(t for t in fig.data if t.type == "scatter3d" and t.mode == "lines" and t.line.width == 6)
    assert lines.line.cmax == points.marker.cmax
    assert list(lines.line.color[:3]) == pytest.approx([maximum, maximum, 0])


def test_displacement_zero_and_unavailable_results():
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    rows = [{"node": n, "Ux_mm": 0, "Uy_mm": 0, "Uz_mm": 0} for n in model.nodes]
    result = SimpleNamespace(ok=True, nodes=pd.DataFrame(rows), elements=pd.DataFrame())
    fig = model_figure(data, model, result, mode="DISPLACEMENT")
    points = next(t for t in fig.data if t.name == "Перемещения узлов")
    assert set(points.marker.color) == {0}
    assert list(points.marker.colorbar.tickvals) == [0]
    assert "максимум 0.000 мм" in fig.layout.title.text
    result.ok = False
    for unavailable in (None, result):
        fig = model_figure(data, model, unavailable, mode="DISPLACEMENT")
        assert "Рассчитать" in fig.layout.annotations[0].text
        assert not any(t.name == "Перемещения узлов" for t in fig.data)
