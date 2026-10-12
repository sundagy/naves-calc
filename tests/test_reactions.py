from src.config import input_from_preset
from src.fem.opensees_solver import VERTICAL_BALANCE_TOLERANCE_KN, solve_model
from src.geometry.truss import generate_model


def test_vertical_reactions_match_vertical_loads():
    data = input_from_preset("original")
    data.roof_dead_load_kpa = 0.25
    model = generate_model(data)
    result = solve_model(data, model)
    assert result.ok, result.message
    lb = result.load_breakdown
    assert abs(lb.snow_load_total_kN - 271.32) < VERTICAL_BALANCE_TOLERANCE_KN
    assert abs(lb.roof_dead_load_total_kN - 25.50) < VERTICAL_BALANCE_TOLERANCE_KN
    assert abs(lb.total_self_weight_kN - result.elements["mass_kg"].sum() * 9.80665 / 1000) < VERTICAL_BALANCE_TOLERANCE_KN
    assert abs(lb.total_self_weight_kN - (
        lb.lower_trusses_self_weight_kN + lb.upper_trusses_self_weight_kN + lb.columns_self_weight_kN
    )) < VERTICAL_BALANCE_TOLERANCE_KN
    assert abs(lb.total_applied_vertical_load_kN - (
        lb.snow_load_total_kN + lb.roof_dead_load_total_kN + lb.total_self_weight_kN
    )) < VERTICAL_BALANCE_TOLERANCE_KN
    assert abs(lb.total_applied_vertical_load_kN - lb.sum_vertical_reactions_kN) < VERTICAL_BALANCE_TOLERANCE_KN
