from src.config import input_from_preset
from src.geometry.canopy import CanopyPlan


def test_plan_shape():
    plan = CanopyPlan()
    assert plan.area == 102.0
    assert [(p.x, p.y) for p in plan.points] == [(0.0, 0.0), (12.0, 0.0), (12.0, 7.0), (0.0, 10.0)]


def test_depth():
    plan = CanopyPlan()
    assert plan.depth(0) == 10
    assert plan.depth(4) == 9
    assert plan.depth(8) == 8
    assert plan.depth(12) == 7


def test_original_preset_counts():
    data = input_from_preset("original")
    assert data.lower_count == 4
    assert data.upper_count == 11

