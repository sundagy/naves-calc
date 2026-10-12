"""Measured model responses, deliberately separate from design acceptance."""
from math import sqrt, pi

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from src.geometry.canopy import roof_plane_from_input
from src.sections.rhs import parse_rhs


def relative_profile(stations, displacement_mm):
    """Remove translation and linear support settlement from a span profile."""
    x, u = np.asarray(stations, dtype=float), np.asarray(displacement_mm, dtype=float)
    if len(x) < 2 or len(x) != len(u) or np.any(np.diff(x) <= 0):
        raise ValueError('Need strictly increasing stations and matching displacements')
    return u - (u[0] + (u[-1] - u[0]) * (x - x[0]) / (x[-1] - x[0]))


def span_responses(model, result):
    """Bottom chord profiles between actual bearing stations; omit cantilevers.

    No universal L/n is assigned: limit and load combination need specification.
    Includes FE subdivision nodes, not just the initial geometry nodes.
    """
    u = result.nodes.set_index('node').Uz_mm
    rows = []
    slaves = {b for _, b in model.bearing_links}
    for group in [*model.lower_trusses, *model.upper_trusses]:
        ids = {n for e in model.elements if e.group == group and 'bottom chord' in e.label for n in (e.i, e.j)}
        if not ids:
            continue
        coords = {n: np.array(model.nodes[n].coords[:2]) for n in ids}
        ordered = sorted(ids, key=lambda n: (coords[n][1], coords[n][0])) if group.startswith('B') else sorted(ids, key=lambda n: coords[n][0])
        origin = coords[ordered[0]]
        direction = coords[ordered[-1]] - origin
        direction /= np.linalg.norm(direction)
        station = {n: float((coords[n] - origin) @ direction) for n in ids}
        ordered = sorted(ids, key=station.get)
        if group.startswith('F'):
            bearings = sorted(ids & slaves, key=station.get)
        else:
            bearings = [n for n in ordered if any(np.linalg.norm(coords[n] - [c.x, c.y]) < 1e-6 for c in model.columns)]
        for a, b in zip(bearings, bearings[1:]):
            chain = [n for n in ordered if station[a] - 1e-8 <= station[n] <= station[b] + 1e-8]
            values = relative_profile([station[n] for n in chain], [u[n] for n in chain])
            idx = int(np.argmax(np.abs(values)))
            length = station[b] - station[a]
            maximum = float(abs(values[idx]))
            rows.append(dict(group=group, support_i=a, support_j=b, L_m=length,
                             node=chain[idx], relative_Uz_mm=float(values[idx]),
                             max_abs_relative_mm=maximum,
                             L_over_f=length * 1000 / maximum if maximum > 1e-10 else None,
                             status='LIMIT_AND_SERVICE_COMBINATION_NOT_ASSIGNED'))
    return pd.DataFrame(rows)


def directional_slopes(xy, z, downhill, reference_xy=None):
    """Piecewise planar surface: positive means drop in nominated direction."""
    xy, z, d = np.asarray(xy), np.asarray(z), np.asarray(downhill)
    d = d / np.linalg.norm(d)
    rows = []
    reference = xy if reference_xy is None else np.asarray(reference_xy)
    for indices in Delaunay(reference).simplices:
        reference_p = reference[indices]
        if abs(np.linalg.det(reference_p[1:] - reference_p[0])) < 1e-8:
            continue
        p = xy[indices]
        matrix = p[1:] - p[0]
        if abs(np.linalg.det(matrix)) < 1e-8:
            continue
        gradient = np.linalg.solve(matrix, z[indices[1:]] - z[indices[0]])
        rows.append(dict(nodes=indices.tolist(), drop_slope=float(-gradient @ d)))
    return rows


def roof_response(data, model, result):
    """Interpolate roof chord surface. Does NOT model substrate sag/drainage."""
    plane = roof_plane_from_input(data)
    downhill = -np.array([plane.a, plane.b])
    if np.linalg.norm(downhill) < 1e-12:
        return pd.DataFrame()
    ids = sorted({n for e in model.elements if e.role == 'upper_chord' and 'top chord' in e.label for n in (e.i, e.j)})
    u = result.nodes.set_index('node')
    # Check surface in displaced plan coordinates, with unscaled displacements.
    xy = [[model.nodes[n].x + u.loc[n, 'Ux_mm']/1000,
           model.nodes[n].y + u.loc[n, 'Uy_mm']/1000] for n in ids]
    z = [model.nodes[n].z + u.loc[n, 'Uz_mm']/1000 for n in ids]
    reference_xy = [model.nodes[n].coords[:2] for n in ids]
    rows = directional_slopes(xy, z, downhill, reference_xy)
    for row in rows:
        row['nodes'] = ','.join(str(ids[i]) for i in row['nodes'])
        row['slope_percent'] = row.pop('drop_slope') * 100
        row['below_0_5_percent'] = row['slope_percent'] < 0.5
    return pd.DataFrame(rows)


def euler_length_thresholds(result, E=2e11):
    """Maximum L_eff at which ideal Euler Ncr equals compression; NOT SP16.

    No FE mesh length used as a buckling length and no approval is inferred.
    """
    rows = []
    for (group, label, section), elements in result.elements.groupby(['group', 'label', 'section']):
        compression = max(0., -elements.N_kN.min())
        if compression <= 1e-8:
            continue
        s = parse_rhs(section)
        rows.append(dict(group=group, label=label, section=section, compression_kN=compression,
                         Euler_Lcrit_weak_m=pi*sqrt(E*min(s.ix, s.iy)/(compression*1000)),
                         status='EFFECTIVE_LENGTH_AND_SP16_CHECK_REQUIRED'))
    return pd.DataFrame(rows)
