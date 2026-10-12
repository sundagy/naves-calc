from dataclasses import replace
from math import pi
import numpy as np
import pytest
from src.models import CanopyInput,Node,Element,Column,StructuralModel
from src.sections.rhs import parse_rhs
from src.geometry.refinement import refine_beams
from src.checks.elastic_stability import assemble_frame
from src.fem.opensees_solver import solve_model


def column_model(offset=0):
    nodes={1:Node(1,0,0,0,'base'),2:Node(2,0,0,3,'top')}
    links=[];roof=2
    if offset:
        nodes[3]=Node(3,offset,0,3,'eccentric');links=[(2,3)];roof=3
    return StructuralModel(nodes,[Element(1,1,2,'column','K','80x40x3','K')],
        [Column('K',0,0,1,2)],{},{},{},[roof],[1],1.,roof_tributary_areas={roof:1.},bearing_links=links)


def test_cantilever_buckling_matches_euler_and_tension_does_not_destabilize():
    data=CanopyInput(spatial_frame=True)
    m=refine_beams(column_model(),8)
    frame=assemble_frame(data,m,{e.id:-1000 for e in m.elements})
    modes=frame.modes(2)
    s=parse_rhs('80x40x3')
    expected=pi*pi*2e11*min(s.ix,s.iy)/(4*3**2)/1000
    assert modes[0][0]==pytest.approx(expected,rel=1e-4)
    assert modes[0][1]<1e-7
    tension=assemble_frame(data,m,{e.id:1000 for e in m.elements})
    q=np.random.default_rng(1).normal(size=tension.G.shape[0])
    assert q@tension.G@q<0


def test_independent_static_assembly_matches_opensees_with_eccentric_joint():
    data=CanopyInput(spatial_frame=True,snow_sg_kpa=0,roof_dead_load_kpa=1)
    m=refine_beams(column_model(.2),4)
    r=solve_model(data,m);assert r.ok
    frame=assemble_frame(data,m,{int(row.element):row.N_kN*1000 for row in r.elements.itertuples()})
    u=frame.solve_loads({int(row.node):[0,0,row.Fz_kN*1000,0,0,0] for row in r.loads.itertuples()})
    expected=r.nodes.set_index('node').loc[frame.node_ids,['Ux_mm','Uy_mm','Uz_mm']].to_numpy()/1000
    assert u[:,:3]==pytest.approx(expected,abs=1e-9)
