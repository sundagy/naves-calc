import pytest

from src.config import input_from_preset
from src.geometry.naves_one import generate_naves_one
from src.geometry.truss import generate_model
from src.visualization.model3d import model_figure


@pytest.mark.parametrize("preset,generate", [("naves_one", generate_naves_one), ("original", generate_model)])
def test_snow_view_applies_vertical_forces_and_conserves_load_without_solver(preset, generate):
    data = input_from_preset(preset)
    data.snow_mu = 1.0
    model = generate(data)
    fig = model_figure(data, model, mode="SNOW_LOAD")
    cones = next(trace for trace in fig.data if trace.type == "cone")
    pressure = data.snow_sg_kpa * data.snow_ce * data.snow_ct * data.snow_gamma_f
    assert sum(row[2] for row in cones.customdata) == pytest.approx(pressure * model.loaded_roof_area)
    assert sum(row[1] for row in cones.customdata) == pytest.approx(model.loaded_roof_area)
    assert cones.anchor == "tip"
    assert all(u == v == 0 and w < 0 for u, v, w in zip(cones.u, cones.v, cones.w))
    for x, y, z, (node_id, area, force) in zip(cones.x, cones.y, cones.z, cones.customdata):
        assert (x, y, z) == model.nodes[node_id].coords
        assert force == pytest.approx(pressure * area)


def test_snow_view_updates_to_zero_without_old_arrows():
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    assert any(trace.type == "cone" for trace in model_figure(data, model, mode="SNOW_LOAD").data)
    data.snow_sg_kpa = 0.0
    fig = model_figure(data, model, mode="SNOW_LOAD")
    assert not any(trace.type == "cone" for trace in fig.data)
    assert "ΣF = 0.00 кН" in fig.layout.title.text
    assert fig.layout.annotations[0].text == "Снеговая нагрузка равна нулю"
