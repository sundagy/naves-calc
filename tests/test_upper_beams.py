from dataclasses import replace

import pytest

from src.config import input_from_preset
from src.fem.assumptions import node_restraints
from src.fem.opensees_solver import solve_model
from src.geometry.naves_one import generate_naves_one


def test_upper_chords_refine_but_webs_do_not_and_bearings_release_rotation():
    data = replace(input_from_preset("naves_one"), spatial_frame=False)
    coarse = generate_naves_one(data)
    fine = generate_naves_one(replace(data, fem_subdivisions=4))
    for role in ("upper_web", "lower_web"):
        assert [e for e in fine.elements if e.role == role] == [e for e in coarse.elements if e.role == role]
    chords = [e for e in fine.elements if e.role == "upper_chord"]
    assert all(e.is_beam for e in chords)
    assert len(chords) == 4 * sum(e.role == "upper_chord" for e in coarse.elements)
    fixed = node_restraints(data, fine)
    upper_nodes = {n for row in fine.upper_bottom_trusses.values() for n in row}
    bearings = [(a, b) for a, b in fine.bearing_links if b in upper_nodes]
    assert bearings
    for a, b in bearings:
        assert a != b
        if fine.nodes[a].coords != fine.nodes[b].coords:
            row = next(row for row in fine.lower_trusses.values() if a in row)
            assert a in (row[0], row[-1])
            assert not fine.nodes[row[0]].y < fine.nodes[b].y < fine.nodes[row[-1]].y
        assert fixed[b][4:] == (0, 0)  # No bending clamp or rigid crossing of chords.


def test_upper_chords_carry_bending_and_mesh_results_converge():
    import openseespy.opensees as ops

    data = replace(input_from_preset("naves_one"), spatial_frame=False)
    deflections = []
    for parts in (1, 2, 4, 8):
        model = generate_naves_one(replace(data, fem_subdivisions=parts))
        result = solve_model(data, model)
        assert result.ok, result.message
        deflections.append(result.max_uz_mm)
        # End bending moments from the actual assembled OpenSees elements.
        moments = [abs(ops.eleResponse(e.id, "localForce")[k])
                   for e in model.elements if e.role == "upper_chord"
                   for k in (4, 5, 10, 11)]
        assert max(moments) > 1  # N*m
        for rows in (model.upper_trusses, model.upper_bottom_trusses):
            for row in rows.values():
                # The torsional datum is unloaded under this planar loading.
                assert ops.nodeReaction(row[0], 4) == pytest.approx(0, abs=1e-6)
        displacements = result.nodes.set_index("node")["Uz_mm"]
        for a, b in model.bearing_links:
            assert displacements[a] == pytest.approx(displacements[b], abs=1e-8)
    # The node attaining the maximum can change with refinement, so successive
    # differences need not decrease monotonically. Require all refined meshes
    # to agree within 0.001 mm instead of comparing two near-zero differences.
    assert max(deflections[1:]) - min(deflections[1:]) < 0.001
