import pytest

from src.config import input_from_preset
from src.geometry.canopy import roof_plane_from_input
from src.loads.snow_sp20 import kaluga_source, mu_for_mono_slope, snow_design_pressure_kpa


def test_kaluga_source_and_pressure_factors():
    data = input_from_preset("original")
    roof = roof_plane_from_input(data)
    mu, s0, sd = snow_design_pressure_kpa(data.snow_sg_kpa, 1.0, 1.0, roof, 1.4)
    assert "Калуга" in kaluga_source()
    assert mu == 1.0
    assert s0 == pytest.approx(1.90)
    assert sd == pytest.approx(2.66)


def test_mu_interpolation():
    assert mu_for_mono_slope(0) == 1.0
    assert mu_for_mono_slope(30) == 1.0
    assert mu_for_mono_slope(60) == 0.0
    assert mu_for_mono_slope(45) == 0.5
