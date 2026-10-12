"""Explicit draft bracing variant for review, never enabled by a preset.

Adds actual beam members and their mass; creates no artificial supports.
All new connections inherit the spatial model's ideal rigid-joint assumption.
Member sizes and attachment details are proposals, not fabrication approval.
"""
from dataclasses import replace
from math import dist
from src.models import Element, Node
from src.geometry.naves_one import UPPER_YS, FRONT_XS, LEFT_REAR_Y, REAR_YS, left_x


def add_proposed_bracing(model, brace_section="60x60x3"):
    if model.mesh_node_ids:
        raise ValueError('Add bracing before finite-element refinement')
    additions=[]
    next_id=max(e.id for e in model.elements)+1
    def add(i,j,label,section=None):
        nonlocal next_id
        if i==j or dist(model.nodes[i].coords,model.nodes[j].coords)<.01:
            raise ValueError('Invalid brace length')
        section=section or brace_section
        group=label.split(':')[0]
        role='lower_bracing' if group in ('lower_plane','lower_tie','vertical_cross') else 'upper_bracing'
        e=Element(next_id,i,j,role,group,section,label,is_beam=True)
        model.elements.append(e);additions.append(e);next_id+=1
    def chord_point(b,y,top):
        label=f'B{b+1} {"top" if top else "bottom"} chord'
        chain=[e for e in model.elements if e.label==label]
        # End attachments follow B when its end frames are inset from columns.
        stations=[model.nodes[n].y for e in chain for n in (e.i,e.j)]
        y=min(max(y,min(stations)),max(stations))
        for e in chain:
            for n in (e.i,e.j):
                if abs(model.nodes[n].y-y)<1e-7:return n
        for e in chain:
            a,z=model.nodes[e.i],model.nodes[e.j]
            t=(y-a.y)/(z.y-a.y)
            if 0<t<1:
                nonlocal next_id
                n=max(model.nodes)+1
                coords=[v+t*(w-v) for v,w in zip(a.coords,z.coords)]
                model.nodes[n]=Node(n,*coords,f'B{b+1}-draft-bottom-{y:g}',{'lower_bottom','proposed_joint'})
                old_j=e.j;e.j=n
                model.elements.append(replace(e,id=next_id,i=n,j=old_j));next_id+=1
                return n
        raise ValueError(f'No chord point {label}, y={y}')
    ends=(LEFT_REAR_Y,*REAR_YS)
    lower,roof={},{}
    for row,y in enumerate(UPPER_YS):
        for b,end in enumerate(ends):
            if y>end+1e-8:continue
            lower[b,row]=chord_point(b,y,True)
            x=left_x(y) if b==0 else FRONT_XS[b]
            roof[b,row]=next(n for n in model.upper_trusses[f'F{row+1}']
                             if abs(model.nodes[n].x-x)<1e-7)
    # Longitudinal runners at roof level connect F top chords into a grid.
    for b in range(5):
        for row in range(9):
            if (b,row+1) in roof:
                add(roof[b,row],roof[b,row+1],f'roof_runner:B{b+1}:F{row+1}-F{row+2}','60x40x3')
    # Horizontal bracing in four strips, at BOTH truss intersection levels.
    # One diagonal per panel: no crossing pipes or artificial crossing nodes.
    for row in (0,4,7,8):
        for b in range(4):
            corners=((b,row),(b+1,row),(b,row+1),(b+1,row+1))
            if not all(p in lower for p in corners):continue
            for grid,name in ((lower,'lower_plane'),(roof,'roof_plane')):
                a,c,d,e=[grid[p] for p in corners]
                add(a,e,f'{name}:B{b+1}-B{b+2}:F{row+1}-F{row+2}:a')
    # Vertical longitudinal diagonals tie roof plane to B top / F bottom plane.
    for b in range(5):
        for row in (0,2,4,6,8):
            if (b,row+1) in lower:
                add(roof[b,row],lower[b,row+1],f'vertical_long:B{b+1}:F{row+1}-F{row+2}')
    # Transverse diaphragms connect the lower B chords at y=0,3,6 m.
    for y in (0.,3.,6.):
        row=UPPER_YS.index(y)
        for b in range(4):
            i,j=chord_point(b,y,False),chord_point(b+1,y,False)
            add(i,j,f'lower_tie:y{y:g}:B{b+1}-B{b+2}')
            add(lower[b,row],j,f'vertical_cross:y{y:g}:B{b+1}-B{b+2}:a')
    model.warnings.append('DRAFT: explicit proposed bracing, all joints ideal rigid; attachment design unverified.')
    return additions
