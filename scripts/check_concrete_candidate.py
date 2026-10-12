"""Compare an explicit draft bracing scheme with unchanged current geometry.

ULS linear rescaling yields 1.05*Gsteel + 1.20*Groof + 1.4*S.
SLS = Gsteel + Groof + S. Neither is a complete project approval.
"""
from pathlib import Path
from dataclasses import replace,asdict
from math import dist
import json
import pandas as pd
import openseespy.opensees as ops
import plotly.graph_objects as go
from src.models import CanopyInput
from src.geometry.naves_one import generate_naves_one
from src.geometry.proposed_bracing import add_proposed_bracing
from src.geometry.refinement import refine_beams
from src.loads.self_weight import element_mass, mass_breakdown
from src.fem.opensees_solver import solve_model
from src.checks.elastic_stability import assemble_frame

OUT=Path('outputs/reports/concrete-candidate');OUT.mkdir(parents=True,exist_ok=True)
BASE=CanopyInput(**json.loads(Path('outputs/reports/frame-600-fixed/input.json').read_text()))
COLORS={'roof_runner':'#008b87','roof_plane':'#dd6c10','lower_plane':'#a427bf','vertical_long':'#1676d2','vertical_cross':'#de4353','lower_tie':'#964229'}


def build(braced):
    m=generate_naves_one(replace(BASE,fem_subdivisions=1))
    added=add_proposed_bracing(m) if braced else []
    return m,added


def render_plot(physical):
    fig=go.Figure()
    for group,color in [('existing','#a9afb7'),*COLORS.items()]:
        xx=[];yy=[];zz=[]
        for e in physical.elements:
            if (group=='existing' and e.group not in COLORS) or (e.group in COLORS and e.group==group):
                for n in (e.i,e.j):
                    a=physical.nodes[n];xx.append(a.x);yy.append(a.y);zz.append(a.z)
                xx.append(None);yy.append(None);zz.append(None)
        fig.add_trace(go.Scatter3d(x=xx,y=yy,z=zz,mode='lines',line=dict(color=color,width=5 if group!='existing' else 2),name={'existing':'Каркас','roof_runner':'Распорки F','roof_plane':'Диагонали кровли','lower_plane':'Диагонали уровня B','vertical_long':'Между уровнями','vertical_cross':'Диафрагмы B','lower_tie':'Распорки B'}[group]))
    fig.update_layout(title='Связи: расчётный черновик' ,height=800,scene=dict(aspectmode='data',xaxis_title='X, м',yaxis_title='Y, м (front=0)',zaxis_title='Z, м'),legend=dict(orientation='h'))
    fig.write_html(OUT/'bracing-3d.html',include_plotlyjs=True)


def main():
    rows=[]
    for variant,braced in [('existing',False),('draft_braced',True)]:
        physical,added=build(braced)
        original=set(physical.nodes)
        if braced:
            pd.DataFrame([dict(id=e.id,i=e.i,j=e.j,label=e.label,section=e.section,
                length_m=dist(physical.nodes[e.i].coords,physical.nodes[e.j].coords),
                mass_kg=element_mass(physical,e)) for e in added]).to_csv(OUT/'proposed-members.csv',index=False)
            (OUT/'physical-model.json').write_text(json.dumps(asdict(physical),ensure_ascii=False,indent=2,default=lambda x:sorted(x)))
            render_plot(physical)
        mass=mass_breakdown(physical)['total']
        added_mass=sum(element_mass(physical,e) for e in added)
        for pattern in ('UNIFORM','LEFT_HALF','RIGHT_HALF','FRONT_HALF','REAR_HALF','SLS_UNIFORM'):
            sls=pattern=='SLS_UNIFORM'
            factor=1. if sls else 1.05
            snow=1. if sls else 1.4
            roof=.45 if sls else .45*1.2
            # All outputs of a linear elastic solution scale by factor.
            # The stiffness and physical density/mass of members are unchanged.
            d=replace(BASE,fem_subdivisions=1,snow_gamma_f=snow/factor,
                      roof_dead_load_kpa=roof/factor,
                      snow_pattern='UNIFORM' if sls else pattern)
            m,_=build(braced)
            for e in m.elements:e.is_beam=dist(m.nodes[e.i].coords,m.nodes[e.j].coords)>=.05
            refine_beams(m,4)
            r=solve_model(d,m)
            if not r.ok:raise RuntimeError(r.message)
            assert r.artificial_reactions.empty or r.artificial_reactions.abs().sum().sum()==0
            ef=r.elements.copy()
            for c in ('N_kN','stress_MPa','utilization','My_max_kNm','Mz_max_kNm','elastic_NM_util'):
                ef[c]*=factor
            ef.to_csv(OUT/f'{variant}-{pattern}-elements.csv',index=False)
            nodes=r.nodes.copy()
            for c in ('Ux_mm','Uy_mm','Uz_mm'):nodes[c]*=factor
            nodes.to_csv(OUT/f'{variant}-{pattern}-nodes.csv',index=False)
            jf=[]
            for e in m.elements:
                f=ops.eleResponse(e.id,'localForce')
                for end,n,k in [('i',e.i,0),('j',e.j,6)]:
                    if n not in original:continue
                    jf.append(dict(label=e.label,role=e.role,section=e.section,node=n,end=end,
                         **{c:f[k+j]*factor/1000 for j,c in enumerate(('N_kN','Vy_kN','Vz_kN','T_kNm','My_kNm','Mz_kNm'))}))
            pd.DataFrame(jf).to_csv(OUT/f'{variant}-{pattern}-joint-demands.csv',index=False)
            worst=ef.loc[ef.elastic_NM_util.idxmax()]
            record=dict(variant=variant,case=pattern,roof_norm_kPa=.45,steel_gamma=factor,
                roof_gamma=1 if sls else 1.2,snow_gamma=snow,mass_kg=mass,added_mass_kg=added_mass,
                Uz_mm=r.max_uz_mm*factor,NM_245=worst.elastic_NM_util,
                NM_230=worst.elastic_NM_util*245/230,worst=worst.label,
                Rz_max_kN=r.reactions.Rz_kN.max()*factor,
                load_kN=r.load_breakdown.total_applied_vertical_load_kN*factor,
                reaction_kN=r.load_breakdown.sum_vertical_reactions_kN*factor)
            if pattern=='UNIFORM':
                frame=assemble_frame(d,m,{int(x.element):x.N_kN*1000 for x in ef.itertuples()})
                modes=frame.modes(1);record['elastic_factor']=modes[0][0]
                record['mode_residual']=modes[0][1]
            rows.append(record)
            print(json.dumps(record,ensure_ascii=False),flush=True)
            pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    (OUT/'assumptions.json').write_text(json.dumps(dict(base_input=asdict(BASE),
        candidate_roof_norm_kPa=.45,ULS='1.05 Gsteel + 1.2 Groof + 1.4 S',
        SLS='Gsteel + Groof + S',steel_Ry_MPa_assumed=245,
        status='DRAFT NOT FOR FABRICATION; rigid joints, welds/local joint strength unverified; roof layers provisional',
        retained_unspecified_original_bracing_mass_kg=251.31),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
