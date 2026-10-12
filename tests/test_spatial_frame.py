from dataclasses import replace

import pytest

from src.config import input_from_preset
from src.fem.assumptions import node_restraints, apply_bearing_constraints
from src.fem.opensees_solver import solve_model
from src.geometry.naves_one import generate_naves_one
from src.geometry.canopy import roof_plane_from_input
from src.materials.steel import SteelMaterial
from src.models import CanopyInput, Column, Element, Node, StructuralModel
from src.sections.rhs import parse_rhs


def test_spatial_preset_has_only_base_supports_and_balances_six_components():
    data = replace(input_from_preset("naves_one"), fem_subdivisions=2)
    assert data.spatial_frame
    assert data.roof_system == "MEMBRANE"
    assert not data.roof_load_confirmed
    assert roof_plane_from_input(data).max_angle_deg == pytest.approx(1.0720051091658)
    model = generate_naves_one(data)
    restraints = node_restraints(data, model)
    assert set(restraints) == set(model.support_node_ids)
    assert len(restraints) == 12
    assert all(v == (1, 1, 1, 1, 1, 1) for v in restraints.values())
    result = solve_model(data, model)
    assert result.ok, result.message
    assert result.artificial_reactions.empty
    assert set(result.elements.formulation) == {"beam"}
    assert (result.equilibrium['Невязка'].abs() < result.equilibrium['Допуск']).all()
    assert result.nodes.Ux_mm.abs().max() > 0.1
    assert not result.reliability_confirmed
    # Joint kinematics, including rotation-induced translation at actual offsets.
    import numpy as np
    import openseespy.opensees as ops
    for master, slave in model.bearing_links:
        a, b = np.array(ops.nodeDisp(master)), np.array(ops.nodeDisp(slave))
        delta = np.array(model.nodes[slave].coords) - model.nodes[master].coords
        assert b[:3] == pytest.approx(a[:3] + np.cross(a[3:], delta), abs=1e-10)
        assert b[3:] == pytest.approx(a[3:], abs=1e-10)


def test_eccentric_vertical_load_transfers_moment_and_horizontal_displacement():
    # Cantilever with a rigid horizontal offset carrying 1 kN vertically.
    # Analytical tip moment M=P*e and u_x=M*L²/(2*E*Iy).
    section, steel = parse_rhs("80x40x3"), SteelMaterial()
    length, offset, force = 3.0, 0.2, 1000.0
    data = CanopyInput(spatial_frame=True, snow_sg_kpa=0, roof_dead_load_kpa=1)
    model = StructuralModel(
        {1: Node(1, 0, 0, 0, "base"), 2: Node(2, 0, 0, length, "top"),
         3: Node(3, offset, 0, length, "loaded eccentric joint")},
        [Element(1, 1, 2, "column", "K", section.name, "K")],
        [Column("K", 0, 0, 1, 2)], {}, {}, {}, [3], [1], 1.0,
        roof_tributary_areas={3: 1.0}, bearing_links=[(2, 3)],
    )
    result = solve_model(data, model)
    assert result.ok
    assert result.reactions.My_kNm.iloc[0] == pytest.approx(-force * offset / 1000)
    nodes = result.nodes.set_index("node")
    assert nodes.loc[2, "Ux_mm"] == pytest.approx(force * offset * length**2 / (2 * steel.E * section.iy) * 1000)
    rotation_y = force * offset * length / (steel.E * section.iy)
    assert nodes.loc[3, "Uz_mm"] - nodes.loc[2, "Uz_mm"] == pytest.approx(-offset * rotation_y * 1000)
    assert result.artificial_reactions.empty


def test_rigid_links_reject_chained_constraints():
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    a, b = model.bearing_links[0]
    model.bearing_links.append((b, a))
    with pytest.raises(ValueError, match="chained"):
        apply_bearing_constraints(None, data, model)
