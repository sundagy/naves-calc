from __future__ import annotations

from dataclasses import replace

from src.models import Node, StructuralModel


def refine_beams(model: StructuralModel, subdivisions: int) -> StructuralModel:
    """Subdivide beams in-place, preserving joints and roof load application points.

    Axial-only truss members remain intact: inserting free internal nodes in
    them would introduce transverse mechanisms. New beam nodes are independent
    of coincident members and carry no additional supports.
    """
    if type(subdivisions) is not int or not 1 <= subdivisions <= 16:
        raise ValueError("FEM subdivisions must be an integer from 1 to 16")
    if subdivisions == 1:
        return model
    node_id = max(model.nodes, default=0) + 1
    element_id = max((e.id for e in model.elements), default=0) + 1
    refined = []
    for element in model.elements:
        if not (element.is_beam or element.role in ("column", "crossbeam")):
            refined.append(element)
            continue
        start, end = model.nodes[element.i], model.nodes[element.j]
        chain = [element.i]
        for step in range(1, subdivisions):
            t = step / subdivisions
            coords = [a + t * (b - a) for a, b in zip(start.coords, end.coords)]
            model.nodes[node_id] = Node(
                node_id, *coords, f"{element.label} / МКЭ {step}", {"mesh"},
            )
            model.mesh_node_ids.add(node_id)
            chain.append(node_id)
            node_id += 1
        chain.append(element.j)
        for step, (i, j) in enumerate(zip(chain, chain[1:])):
            refined.append(replace(element, id=element.id if step == 0 else element_id, i=i, j=j))
            if step:
                element_id += 1
    model.elements = refined
    return model
