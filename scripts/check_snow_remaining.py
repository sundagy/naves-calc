"""Snow-only diagnostic for 600 mm / 80x80x3 model; fixed base hypothesis.
Not a full SP16 or weld design check. Saves all simultaneous end-force demands.
"""
import argparse,json
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
import openseespy.opensees as ops
from src.models import CanopyInput
from src.geometry.naves_one import generate_naves_one
from src.geometry.refinement import refine_beams
from src.fem.opensees_solver import solve_model
from src.checks.elastic_stability import assemble_frame

parser=argparse.ArgumentParser()
parser.add_argument('--mesh',type=int,default=4)
parser.add_argument('--roof-load',type=float,default=.15)
parser.add_argument('--pattern',default='UNIFORM')
args=parser.parse_args()
out=Path('outputs/reports/snow-completion');out.mkdir(parents=True,exist_ok=True)
d=CanopyInput(**json.loads(Path('outputs/reports/frame-600-fixed/input.json').read_text()))
d=replace(d,fem_subdivisions=1,roof_dead_load_kpa=args.roof_load,snow_pattern=args.pattern)
m=generate_naves_one(d)
physical_ids={n for e in m.elements for n in (e.i,e.j)}
# Mesh ALL members for flexural buckling, including truss webs.
from math import dist
for e in m.elements:
    # Do not subdivide already tiny bearing segments; they remain beam elements.
    e.is_beam=dist(m.nodes[e.i].coords,m.nodes[e.j].coords)>=.05
m=refine_beams(m,args.mesh)
r=solve_model(d,m)
if not r.ok:raise RuntimeError(r.message)
tag=f'{args.pattern}-q{args.roof_load:g}-mesh{args.mesh}'
# Capture all six concurrent local actions at ORIGINAL joints; no envelope mixing.
forces=[]
for e in m.elements:
    local=ops.eleResponse(e.id,'localForce')
    for end,n,start in [('i',e.i,0),('j',e.j,6)]:
        if n not in physical_ids:continue
        forces.append(dict(element=e.id,label=e.label,role=e.role,node=n,end=end,section=e.section,
            **{key:local[start+j]/1000 for j,key in enumerate(['N_kN','Vy_kN','Vz_kN','T_kNm','My_kNm','Mz_kNm'])}))
pd.DataFrame(forces).to_csv(out/f'{tag}-joint-demands.csv',index=False)
r.elements.to_csv(out/f'{tag}-elements.csv',index=False)
frame=assemble_frame(d,m,{int(row.element):row.N_kN*1000 for row in r.elements.itertuples()})
u=frame.solve_loads({int(row.node):[0,0,row.Fz_kN*1000,0,0,0] for row in r.loads.itertuples()})
expected=r.nodes.set_index('node').loc[frame.node_ids,['Ux_mm','Uy_mm','Uz_mm']].to_numpy()/1000
error=float(np.max(np.abs(u[:,:3]-expected)))
# 0.01% displacement agreement; tiny bearing segments make K ill-conditioned.
tolerance=1e-9+1e-4*float(np.max(np.abs(expected)))
if error>tolerance:raise RuntimeError(f'Independent K disagrees with OpenSees: {error} m, tolerance={tolerance}')
print('Static assembly verified',tag,'error_m',error,flush=True)
modes=frame.modes(3)
mode_rows=[]
for j,(factor,residual,shape) in enumerate(modes):
    idx=np.argsort(np.linalg.norm(shape[:,:3],axis=1))[-10:][::-1]
    mode_rows.append(dict(mode=j+1,factor=factor,residual=residual,
        peak_nodes=[m.nodes[frame.node_ids[i]].label for i in idx[:5]]))
    pd.DataFrame(dict(node=frame.node_ids,Ux=shape[:,0],Uy=shape[:,1],Uz=shape[:,2])).to_csv(out/f'{tag}-mode{j+1}.csv',index=False)
worst=r.elements.loc[r.elements.elastic_NM_util.idxmax()]
summary=dict(case=args.pattern,roof_load=args.roof_load,mesh=args.mesh,Uz_mm=r.max_uz_mm,
             NM_max=float(worst.elastic_NM_util),worst=worst.label,static_error_m=error,modes=mode_rows,
             limitations='Elastic flexural initial-stress eigenproblem. All reference loads scale together. Not snow capacity; no SP16 acceptance, local wall buckling, yielding, imperfections, torsional initial-stress terms or weld flexibility. All members subdivided; selfweight lumping changes with mesh.')
(out/f'{tag}-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
