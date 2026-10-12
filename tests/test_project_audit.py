import numpy as np
import pytest
from src.checks.project_audit import relative_profile, directional_slopes


def test_relative_profile_removes_support_settlement_but_keeps_bending():
    assert relative_profile([0, 2, 4], [-10, -17, -20]) == pytest.approx([0, -2, 0])
    assert relative_profile([0, 2, 4], [0, -2, 0]) == pytest.approx([0, -2, 0])
    with pytest.raises(ValueError):
        relative_profile([0, 0], [0, 0])


def test_surface_slope_includes_direction_and_real_units():
    xy = np.array([[0., 0.], [2., 0.], [0., 2.], [2., 2.]])
    z = 3 - .02 * xy[:, 0] + .01 * xy[:, 1]
    direction = [.02, -.01]
    rows = directional_slopes(xy, z, direction)
    assert len(rows) == 2
    assert [r['drop_slope'] for r in rows] == pytest.approx([np.hypot(.02, .01)] * 2)
    reverse = directional_slopes(xy, -z, direction)
    assert all(r['drop_slope'] < 0 for r in reverse)


def test_roof_mesh_connectivity_does_not_create_slivers_along_bent_chord():
    reference = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.], [2., 1.]])
    displaced = reference.copy()
    displaced[1, 1] = -.00001
    z = 3 - .02 * displaced[:, 0]
    rows = directional_slopes(displaced, z, [1, 0], reference)
    assert len(rows) == 4
    assert [r['drop_slope'] for r in rows] == pytest.approx([.02]*4)
