from dataclasses import replace

import pytest

from src.config import input_from_preset
from src.fem.opensees_solver import solve_model
from src.geometry.naves_one import generate_naves_one
from src.geometry.refinement import refine_beams
from src.geometry.truss import validate_connectivity
from src.loads.self_weight import G_ACCEL, mass_breakdown
from src.loads.tributary import tributary_roof_loads
from src.materials.steel import SteelMaterial
from src.models import CanopyInput, Column, Element, Node, StructuralModel
from src.sections.database import section_database


@pytest.mark.parametrize("parts", [2, 4, 8, 16])
def test_refined_canopy_preserves_structure_and_load_balance(parts):
    data = input_from_preset("naves_one")
    original = generate_naves_one(data)
    data = replace(data, fem_subdivisions=parts)
    model = generate_naves_one(data)
    beams = sum(e.is_beam or e.role in ("column", "crossbeam") for e in original.elements)
    assert len(model.elements) == len(original.elements) + beams * (parts - 1)
    assert len(model.nodes) == len(original.nodes) + beams * (parts - 1)
    assert len({e.id for e in model.elements}) == len(model.elements)
    assert model.bearing_links == original.bearing_links
    assert model.support_node_ids == original.support_node_ids
    assert mass_breakdown(model) == pytest.approx(mass_breakdown(original))
    assert tributary_roof_loads(data, model, 2) == tributary_roof_loads(data, original, 2)
    validate_connectivity(model)
    result = solve_model(data, model)
    assert result.ok
    lb = result.load_breakdown
    assert lb.sum_vertical_reactions_kN == pytest.approx(lb.total_applied_vertical_load_kN, abs=1e-6)
    assert len(result.nodes) == len(model.nodes)
    assert len(result.elements) == len(model.elements)


@pytest.mark.parametrize("end", [(4, 0, 0), (0, 4, 0), (2.4, 3.2, 0)])
def test_mesh_nodes_do_not_clamp_beam_and_deflection_converges(end):
    # Fixed-ended horizontal beam under its own distributed weight.
    # With clamped internal rotations, refinement would drive deflection to zero.
    data = CanopyInput(roof_dead_load_kpa=0, snow_sg_kpa=0)
    steel = SteelMaterial()
    section = section_database()["80x40x3"]
    length = 4.0
    expected = section.area * steel.rho * G_ACCEL * length**4 / (384 * steel.E * section.ix) * 1000
    errors = []
    for parts in (4, 8, 16):
        model = StructuralModel(
            nodes={1: Node(1, 0, 0, 0, "A"), 2: Node(2, *end, "B")},
            elements=[Element(1, 1, 2, "crossbeam", "test", "80x40x3", "Beam")],
            columns=[Column("A", 0, 0, 1, 1), Column("B", length, 0, 2, 2)],
            lower_trusses={}, upper_trusses={}, upper_bottom_trusses={},
            roof_node_ids=[], support_node_ids=[1, 2], loaded_roof_area=0,
            roof_tributary_areas={},
        )
        result = solve_model(data, refine_beams(model, parts))
        assert result.ok
        errors.append(abs(-result.max_uz_mm - expected))
    assert errors[2] <= errors[1] + 1e-9
    assert errors[1] <= errors[0] + 1e-9
    assert errors[-1] / expected < 0.02


@pytest.mark.parametrize("parts", [0, 17, 2.5, True])
def test_invalid_subdivisions(parts):
    data = replace(input_from_preset("naves_one"), fem_subdivisions=parts)
    with pytest.raises(ValueError, match="subdivisions"):
        generate_naves_one(data)
