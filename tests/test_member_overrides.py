from dataclasses import replace
import pytest
from src.config import input_from_preset
from src.geometry.naves_one import generate_naves_one
from src.loads.self_weight import mass_breakdown


def test_whole_chord_override_preserves_geometry_and_changes_only_target_mass():
    data = replace(input_from_preset('naves_one'), fem_subdivisions=4)
    baseline = generate_naves_one(data)
    changed = generate_naves_one(replace(data,member_section_overrides={'B3 top chord':'60x60x4'}))
    assert baseline.nodes == changed.nodes
    assert baseline.bearing_links == changed.bearing_links
    assert baseline.roof_tributary_areas == changed.roof_tributary_areas
    for a,b in zip(baseline.elements,changed.elements):
        assert (a.id,a.i,a.j,a.label)==(b.id,b.i,b.j,b.label)
        assert b.section == ('60x60x4' if a.label=='B3 top chord' else a.section)
    assert mass_breakdown(changed)['total'] > mass_breakdown(baseline)['total']


@pytest.mark.parametrize('overrides, message',[
    ({'B33 top chord':'60x60x4'},'Unknown member'),
    ({'B3 top chord':'60x60x9'},'Unknown override'),
    ({'B3 top chord':'80x80x3'},'outside dimensions'),
])
def test_invalid_override_rejected(overrides,message):
    with pytest.raises(ValueError,match=message):
        generate_naves_one(replace(input_from_preset('naves_one'),member_section_overrides=overrides))
