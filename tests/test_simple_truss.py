import pytest


def test_simple_opensees_axial_bar():
    ops = pytest.importorskip("openseespy.opensees")
    E = 2.0e11
    A = 1.0e-3
    L = 2.0
    F = 10_000.0
    ops.wipe()
    ops.model("BasicBuilder", "-ndm", 3, "-ndf", 3)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, L, 0.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 1)
    ops.uniaxialMaterial("Elastic", 1, E)
    ops.element("truss", 1, 1, 2, A, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, F, 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    ops.reactions()
    assert ops.nodeReaction(1)[0] == pytest.approx(-F)
    assert ops.nodeDisp(2)[0] == pytest.approx(F * L / (E * A))
    ops.wipe()

