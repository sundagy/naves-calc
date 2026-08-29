from src.config import input_from_preset
from src.fem.opensees_solver import solve_model
from src.geometry.truss import generate_model


def test_vertical_reactions_match_vertical_loads():
    data = input_from_preset("original")
    model = generate_model(data)
    result = solve_model(data, model)
    assert result.ok, result.message
    applied = result.load_breakdown.total_roof_load_n + result.load_breakdown.total_self_weight_n
    assert abs(result.total_reaction_z_n - applied) / applied < 0.001

