"""Research elastic flexural buckling of the ideal rigid-jointed frame.

Euler-Bernoulli elastic K and consistent initial-stress Kg from axial forces.
No shell/local buckling, yielding, weld flexibility, imperfections or SP16 approval.
Eigenvalue scales ALL reference loads together; it is not extra snow capacity.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu, eigsh, LinearOperator
from src.fem.assumptions import beam_width_axis
from src.sections.rhs import parse_rhs


def local_matrices(section, L, compression, E=2e11, nu=.3):
    k, g = np.zeros((12,12)), np.zeros((12,12))
    for ids, value in [([0,6],E*section.area/L),([3,9],E/(2*(1+nu))*section.torsion_j/L)]:
        k[np.ix_(ids,ids)] += value*np.array([[1,-1],[-1,1]])
    bend = np.array([[12,6*L,-12,6*L],[6*L,4*L*L,-6*L,2*L*L],
                     [-12,-6*L,12,-6*L],[6*L,2*L*L,-6*L,4*L*L]])
    geo = np.array([[36,3*L,-36,3*L],[3*L,4*L*L,-3*L,-L*L],
                    [-36,-3*L,36,-3*L],[3*L,-L*L,-3*L,4*L*L]])
    for ids,I,sign in [([1,5,7,11],section.ix,[1,1,1,1]),([2,4,8,10],section.iy,[1,-1,1,-1])]:
        signs=np.outer(sign,sign)
        k[np.ix_(ids,ids)] += E*I/L**3 * bend*signs
        g[np.ix_(ids,ids)] += compression/(30*L) * geo*signs
    return k,g


@dataclass
class FrameStability:
    K: object
    G: object
    C: object
    node_ids: list

    def solve_loads(self, loads):
        f=np.zeros(len(self.node_ids)*6)
        index={n:i for i,n in enumerate(self.node_ids)}
        for n,force in loads.items():
            f[6*index[n]:6*index[n]+6] += force
        ur=splu(self.K.tocsc()).solve(np.asarray(self.C.T@f).ravel())
        return np.asarray(self.C@ur).reshape((-1,6))

    def modes(self, count=3):
        scaling=1/np.sqrt(self.K.diagonal())
        D=diags(scaling)
        k=(D@self.K@D).tocsc();g=(D@self.G@D).tocsc()
        lu=splu(k)
        inverse=LinearOperator(k.shape,matvec=lu.solve)
        mu,v=eigsh(g,k=count,M=k,Minv=inverse,which='LA',tol=1e-8,
                   v0=np.random.default_rng(0).normal(size=k.shape[0]),maxiter=5000)
        rows=[]
        for j in np.argsort(mu)[::-1]:
            if mu[j]<=0:
                continue
            factor=1/mu[j];q=scaling*v[:,j]
            a=self.K@q;b=factor*(self.G@q)
            residual=np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b))
            if residual>1e-5:
                raise ValueError(f'Unconverged buckling mode: {residual}')
            full=np.asarray(self.C@q).reshape((-1,6))
            full/=np.max(np.linalg.norm(full[:,:3],axis=1))
            rows.append((float(factor),float(residual),full))
        return rows


def assemble_frame(data,model,axial_n):
    if not data.spatial_frame or data.base_fixity != 'FIXED':
        raise ValueError('Diagnostic requires spatial frame and fixed bases')
    nodes=sorted(model.nodes);index={n:i for i,n in enumerate(nodes)}
    slaves={b:a for a,b in model.bearing_links}
    if set(slaves)&set(slaves.values()) or len(slaves)!=len(model.bearing_links):
        raise ValueError('Chained or duplicate rigid constraints')
    base=set(model.support_node_ids)
    if base&set(slaves):
        raise ValueError('Slave support')
    independent=[n for n in nodes if n not in slaves and n not in base]
    ridx={n:i for i,n in enumerate(independent)}
    cr,cc,cv=[],[],[]
    for n in nodes:
        master=slaves.get(n,n)
        if master not in ridx:
            continue
        block=np.eye(6)
        if n in slaves:
            x,y,z=np.array(model.nodes[n].coords)-model.nodes[master].coords
            block[:3,3:]=np.array([[0,z,-y],[-z,0,x],[y,-x,0]])
        ii,jj=np.nonzero(block)
        cr.extend(6*index[n]+ii);cc.extend(6*ridx[master]+jj);cv.extend(block[ii,jj])
    C=coo_matrix((cv,(cr,cc)),shape=(6*len(nodes),6*len(independent))).tocsr()
    rr,cc,kv,gv=[],[],[],[]
    for e in model.elements:
        delta=np.array(model.nodes[e.j].coords)-model.nodes[e.i].coords
        L=np.linalg.norm(delta);x=delta/L
        y=np.cross(beam_width_axis(model,e),x);y/=np.linalg.norm(y)
        z=np.cross(x,y);rotation=np.array([x,y,z])
        T=np.kron(np.eye(4),rotation)
        k,g=local_matrices(parse_rhs(e.section),L,-axial_n[e.id])
        k=T.T@k@T;g=T.T@g@T
        ids=np.r_[np.arange(6*index[e.i],6*index[e.i]+6),np.arange(6*index[e.j],6*index[e.j]+6)]
        rr.extend(np.repeat(ids,12));cc.extend(np.tile(ids,12));kv.extend(k.ravel());gv.extend(g.ravel())
    shape=(6*len(nodes),)*2
    K=C.T@coo_matrix((kv,(rr,cc)),shape=shape).tocsr()@C
    G=C.T@coo_matrix((gv,(rr,cc)),shape=shape).tocsr()@C
    return FrameStability(((K+K.T)*.5).tocsc(),((G+G.T)*.5).tocsc(),C,nodes)
