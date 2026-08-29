from src.config import input_from_preset
from src.geometry.truss import generate_model
from src.loads.tributary import tributary_roof_loads


def test_uniform_snow_load_is_conserved():
    data = input_from_preset("original")
    model = generate_model(data)
    pressure_kpa = 2.66
    loads = tributary_roof_loads(data, model, pressure_kpa)
    expected = pressure_kpa * 1000.0 * 102.0
    assert abs(sum(loads.values()) + expected) / expected < 0.001

