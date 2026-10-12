from math import dist
from dataclasses import replace
import pytest
from src.models import CanopyInput
from src.geometry.naves_one import generate_naves_one
from src.geometry.proposed_bracing import add_proposed_bracing
from src.geometry.truss import validate_connectivity
from src.loads.self_weight import mass_breakdown,element_mass
from scripts.check_candidate_members import central_c,eccentric_phi,PHI,LAMS,MS


def test_real_braces_preserve_supports_load_area_and_mass_balance():
    data=CanopyInput(preset_name='naves_one',lower_height_mm=600,
        lower_chord_section='80x80x3',lower_web_section='60x60x3',
        upper_chord_section='60x60x3',upper_web_section='40x40x3',
        upper_height_mm=400,upper_end_height_mm=170,upper_max_diagonal_mm=800,
        column_section='120x120x3',column_height_mm=2700,spatial_frame=True)
    model=generate_naves_one(data)
    supports=list(model.support_node_ids);areas=dict(model.roof_tributary_areas)
    old_mass=mass_breakdown(model)['total']
    added=add_proposed_bracing(model)
    validate_connectivity(model)
    assert model.support_node_ids==supports
    assert model.roof_tributary_areas==areas
    assert all(dist(model.nodes[e.i].coords,model.nodes[e.j].coords)>0 for e in model.elements)
    assert len({e.id for e in model.elements})==len(model.elements)
    mass=mass_breakdown(model)
    assert mass['total']==pytest.approx(old_mass+sum(element_mass(model,e) for e in added))
    assert mass['total']==pytest.approx(sum(mass[k] for k in ('lower','upper','columns','crossbeams')))
    assert not set(model.support_node_ids)&{n for e in added for n in (e.i,e.j)}


def test_sp16_reference_values_and_no_extrapolation():
    # Published D.1 curve c and D.3 independent reference points.
    assert central_c(2)==pytest.approx(.744,abs=.001)
    assert central_c(4)==pytest.approx(.402,abs=.001)
    assert PHI[3, list(MS).index(5)]==pytest.approx(.222)
    assert PHI[10,list(MS).index(20)]==pytest.approx(.051)
    assert eccentric_phi(2,0)<=central_c(2)
    with pytest.raises(ValueError):eccentric_phi(6,1)
    with pytest.raises(ValueError):eccentric_phi(2,30)
