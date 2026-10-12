from dataclasses import replace

import pytest

from src.config import input_from_preset
from src.fem.assumptions import beam_width_axis, node_restraints
from src.fem.opensees_solver import solve_model
from src.geometry.naves_one import generate_naves_one
from src.materials.steel import SteelMaterial
from src.models import BaseFixity, Element, Node, StructuralModel
from src.sections.rhs import parse_rhs


@pytest.mark.parametrize("direction", [0, 1])
def test_rectangular_column_bends_about_correct_axis(direction):
    import openseespy.opensees as ops

    length, force = 3.0, 1000.0
    section, steel = parse_rhs("80x40x3"), SteelMaterial()
    element = Element(1, 1, 2, "column", "K", section.name, "K")
    model = StructuralModel(
        {1: Node(1, 0, 0, 0, "base"), 2: Node(2, 0, 0, length, "top")},
        [element], [], {}, {}, {}, [], [1], 0,
    )
    ops.wipe()
    try:
        ops.model("BasicBuilder", "-ndm", 3, "-ndf", 6)
        for node in model.nodes.values():
            ops.node(node.id, *node.coords)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.geomTransf("Linear", 1, *beam_width_axis(model, element))
        ops.element("elasticBeamColumn", 1, 1, 2, section.area, steel.E, steel.G,
                    section.torsion_j, section.iy, section.ix, 1)
        load = [0.0] * 6
        load[direction] = force
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, *load)
        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1)
        ops.algorithm("Linear")
        ops.analysis("Static")
        assert ops.analyze(1) == 0
        inertia = section.iy if direction == 0 else section.ix
        assert ops.nodeDisp(2, direction + 1) == pytest.approx(force * length**3 / (3 * steel.E * inertia))
    finally:
        ops.wipe()


def test_artificial_restraints_are_reported_separately_from_bases():
    data = replace(input_from_preset("naves_one"), fem_subdivisions=4, spatial_frame=False)
    model = generate_naves_one(data)
    fixed = node_restraints(data, model)
    assert model.mesh_node_ids.isdisjoint(fixed)
    assert all(fixed[n] == (1, 1, 1, 1, 1, 1) for n in model.support_node_ids)
    pinned = node_restraints(replace(data, base_fixity=BaseFixity.PINNED), model)
    assert all(pinned[n] == (1, 1, 1, 0, 0, 0) for n in model.support_node_ids)
    result = solve_model(data, model)
    assert result.ok
    artificial = result.artificial_reactions
    assert set(artificial.node) == set(fixed) - set(model.support_node_ids)
    assert artificial[["Rx_kN", "Ry_kN"]].abs().max().max() > 1
    # No horizontal external loads: base and artificial reactions balance together.
    for component in ("Rx_kN", "Ry_kN"):
        assert artificial[component].sum() + result.reactions[component].sum() == pytest.approx(0, abs=1e-6)
