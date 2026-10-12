"""Conditional SP16 9.2.10 audit for the DRAFT restrained geometry.
Effective lengths are design hypotheses requiring restraint stiffness/joint checks.
Not a complete SP16 implementation or local/weld approval.
"""
from pathlib import Path
from math import sqrt,dist,pi
import numpy as np
import pandas as pd
from scripts.check_concrete_candidate import build,OUT
from src.sections.rhs import parse_rhs
from src.geometry.naves_one import UPPER_YS,FRONT_XS,left_x

LAMS=np.arange(.5,5.51,.5)
MS=np.array([.1,.25,.5,.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,8,9,10,12,14,17,20])
# SP16 D.3, rows 0.5..5.5, read against published table with amendments 1-6.
PHI=np.array([
[967,922,850,782,722,669,620,577,538,469,417,370,337,307,280,260,237,222,210,183,164,150,125,106,90,77],
[925,854,778,711,653,600,563,520,484,427,382,341,307,283,259,240,225,209,196,175,157,142,121,103,86,74],
[875,804,716,647,593,548,507,470,439,388,347,312,283,262,240,223,207,195,182,163,148,134,114,99,82,70],
[813,742,653,587,536,496,457,425,397,352,315,286,260,240,222,206,193,182,170,153,138,125,107,94,79,67],
[742,672,587,526,480,442,410,383,357,317,287,262,238,220,204,190,178,168,158,144,130,118,101,90,76,65],
[667,597,520,465,425,395,365,342,320,287,260,238,217,202,187,175,166,156,147,135,123,112,97,86,73,63],
[587,522,455,408,375,350,325,303,287,258,233,216,198,183,172,162,153,145,137,125,115,106,92,82,69,60],
[505,447,394,356,330,309,289,270,256,232,212,197,181,168,158,149,140,135,127,118,108,98,88,78,66,57],
[418,382,342,310,288,272,257,242,229,208,192,178,165,155,146,137,130,125,118,110,101,93,83,75,64,55],
[354,326,295,273,253,239,225,215,205,188,175,162,150,143,135,126,120,117,111,103,95,88,79,72,62,53],
[302,280,256,240,224,212,200,192,184,170,158,148,138,132,124,117,112,108,104,95,89,84,75,69,60,51]])/1000


def central_c(lam):
    if lam==0:return 1.
    delta=9.87*(1-.04+.14*lam)+lam*lam
    return min(1.,19.74/(delta+sqrt(delta*delta-39.48*lam*lam)),7.6/(lam*lam))


def eccentric_phi(lam,m):
    eta=1.1 if lam>5 or m>5 else 1.35-.05*m-.01*(5-m)*lam
    mef=m*eta
    if lam>5.5 or mef>20:raise ValueError('Outside transcribed D.3 domain')
    tab=float(np.interp(max(.1,mef),MS,[np.interp(max(.5,lam),LAMS,PHI[:,k]) for k in range(len(MS))]))
    return min(tab,central_c(lam))


def check(section,N,Mx,My,L,Ry=245e6):
    s=parse_rhs(section);A=s.area
    lx=L/s.rx*sqrt(Ry/2e11);ly=L/s.ry*sqrt(Ry/2e11)
    mx=Mx*A/(N*s.wx);my=My*A/(N*s.wy)
    dx=1 if lx<=1 else 1-.1*N*lx*lx/(A*Ry)
    dy=1 if ly<=1 else 1-.1*N*ly*ly/(A*Ry)
    if min(dx,dy)<=0:raise ValueError('Nonpositive second-order denominator')
    ux=N/(eccentric_phi(ly,my)*A*Ry)+Mx/(dx*s.wx*Ry)
    uy=N/(eccentric_phi(lx,mx)*A*Ry)+My/(dy*s.wy*Ry)
    return max(ux,uy),max(lx,ly),max(mx,my)


def length_for(model,label,es):
    maxL=max(dist(model.nodes[e.i].coords,model.nodes[e.j].coords) for e in es)
    role=es[0].role
    if role=='column':return 2*maxL,'2H, fixed-free conservative member hypothesis'
    if role=='lower_chord':
        b=int(label[1])-1
        ids={n for e in es for n in (e.i,e.j)}
        ys=sorted({model.nodes[n].y for n in ids});end=ys[-1]
        if 'top' in label:marks=[0.,*[y for y in UPPER_YS if 0<y<end],end]
        else:marks=sorted({0.,3.,6.,end,*[model.nodes[c.top_node].y for c in model.columns if abs(model.nodes[c.top_node].x-(left_x(model.nodes[c.top_node].y) if b==0 else FRONT_XS[b]))<1e-6 and model.nodes[c.top_node].y<=end]})
        factor=sqrt(1+(2.588/9.659)**2) if b==0 else 1.
        return max(maxL,max(np.diff(marks))*factor),'proposed lateral restraint spacing, both axes conservatively equal'
    if role=='upper_chord':
        ids={n for e in es for n in (e.i,e.j)};xs=sorted({model.nodes[n].x for n in ids})
        y=model.nodes[es[0].i].y
        marks=[x for x in (left_x(y),*FRONT_XS[1:]) if xs[0]-1e-7<=x<=xs[-1]+1e-7]
        spans=list(np.diff(marks))+[2*(marks[0]-xs[0]),2*(xs[-1]-marks[-1])]
        return max(maxL,max(spans))*1.001,'B crossings tied by proposed bracing; cantilever length doubled'
    return maxL,'pin-ended physical member length; not FE length'


def main():
    model,_=build(True);bylabel={}
    for e in model.elements:bylabel.setdefault(e.label,[]).append(e)
    rows=[]
    for p in sorted(OUT.glob('draft_braced-*-elements.csv')):
        if 'SLS' in p.name:continue
        df=pd.read_csv(p);case=p.name.removeprefix('draft_braced-').removesuffix('-elements.csv')
        for label,g in df.groupby('label'):
            N=max(0.,-g.N_kN.min())*1000
            if N<1e-6:continue
            L,basis=length_for(model,label,bylabel[label]);section=g.section.iloc[0]
            Mx=g.Mz_max_kNm.max()*1000;My=g.My_max_kNm.max()*1000
            s=parse_rhs(section);euler=N*L*L/(pi*pi*2e11*min(s.ix,s.iy))
            row=dict(case=case,label=label,section=section,L_hypothesis_m=L,basis=basis,
                     Ncomp_kN=N/1000,Mx_envelope_kNm=Mx/1000,My_envelope_kNm=My/1000,Euler_ratio=euler)
            try:
                u,lam,m=check(section,N,Mx,My,L)
                row.update(util_9_2_10=u,lambda_bar=lam,m=m,status='conditional_pass' if u<=1 else 'conditional_exceedance')
            except ValueError as exc:row.update(status='not_evaluated',reason=str(exc))
            rows.append(row)
    allrows=pd.DataFrame(rows);allrows.to_csv(OUT/'conditional-member-checks.csv',index=False)
    print(allrows.groupby('status').size())
    print(allrows.sort_values('util_9_2_10',ascending=False)[['case','label','L_hypothesis_m','util_9_2_10']].head(12).to_string(index=False))

if __name__=='__main__':main()
