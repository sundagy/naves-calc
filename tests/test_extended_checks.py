from dataclasses import replace

import pytest

from src.checks.member import member_screening
from src.config import input_from_preset
from src.fem.assumptions import node_restraints
from src.fem.opensees_solver import solve_model
from src.geometry.naves_one import generate_naves_one
from src.geometry.refinement import refine_beams
from src.loads.scenarios import snow_pattern_loads
from src.materials.steel import SteelMaterial
from src.models import CanopyInput, Column, Element, Node, StructuralModel
from src.sections.rhs import parse_rhs


@pytest.mark.parametrize("pair", [("LEFT_HALF", "RIGHT_HALF"), ("FRONT_HALF", "REAR_HALF")])
def test_half_snow_integrates_cells_and_recombines_to_uniform(pair):
    data = input_from_preset("naves_one")
    model = generate_naves_one(data)
    full = snow_pattern_loads(data, model, 2.66)
    left, right = [snow_pattern_loads(replace(data, snow_pattern=p), model, 2.66) for p in pair]
    for node in full:
        assert left[node] + right[node] == pytest.approx(full[node], abs=1e-8)
    assert -sum(full.values()) == pytest.approx(2660 * model.loaded_roof_area)
    assert 0 < -sum(left.values()) < -sum(full.values())


@pytest.mark.parametrize("subdivisions", [1, 4])
def test_cantilever_bending_reactions_and_euler_length_independent_of_mesh(subdivisions):
    section, steel = parse_rhs("80x40x3"), SteelMaterial()
    length = 2.0
    data = CanopyInput(snow_sg_kpa=0, roof_dead_load_kpa=0,
                       spatial_diagnostic=True, effective_lengths_m={"beam": 4.0})
    model = StructuralModel(
        {1: Node(1, 0, 0, 3, "fixed"), 2: Node(2, length, 0, 3, "tip")},
        [Element(1, 1, 2, "crossbeam", "test", section.name, "beam")],
        [Column("base", 0, 0, 1, 2)], {}, {}, {}, [], [1], 0,
        roof_tributary_areas={},
    )
    refine_beams(model, subdivisions)
    result = solve_model(data, model)
    assert result.ok
    weight = section.area * steel.rho * 9.80665 * length
    # End-lumped self weight: moment about the fixed end is w L²/2 for any mesh.
    moment = weight * length / 2
    assert result.reactions.My_kNm.iloc[0] == pytest.approx(-moment / 1000)
    assert result.elements.Mz_max_kNm.max() == pytest.approx(moment / 1000)
    assert result.elements.My_max_kNm.max() == pytest.approx(0, abs=1e-10)
    assert result.elements.elastic_NM_util.max() == pytest.approx(moment / section.wx / steel.resistance_pa)
    if subdivisions == 1:
        assert result.max_uz_mm == pytest.approx(-weight / 2 * length**3 / (3 * steel.E * section.ix) * 1000)
    assert set(result.elements.effective_length_m) == {4.0}
    assert not result.reliability_confirmed
    assert not result.checks.empty


def test_spatial_diagnostic_never_clamps_chords_or_horizontal_translations():
    data = replace(input_from_preset("naves_one"), spatial_diagnostic=True, spatial_frame=False)
    model = generate_naves_one(data)
    restraints = node_restraints(data, model)
    for node, flags in restraints.items():
        if node not in model.support_node_ids:
            assert flags[:3] == (0, 0, 0)
    beams = {n for e in model.elements if e.is_beam or e.role in ("column", "crossbeam") for n in (e.i, e.j)}
    assert (beams - set(model.support_node_ids)).isdisjoint(restraints)


def test_euler_unknown_length_is_not_silently_taken_from_mesh():
    section, steel = parse_rhs("40x40x2"), SteelMaterial()
    result = member_screening(section, steel, -34000, None)
    assert result["Euler_ratio"] is None
    assert result["effective_length_m"] is None
    result = member_screening(section, steel, -34000, None, 1.1)
    assert result["Euler_ratio"] > 0
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            member_screening(section, steel, -34000, None, invalid)
