from src.sections.rhs import parse_rhs
import pytest


def test_rhs_properties():
    section = parse_rhs("60x40x3")
    expected_area = 0.060 * 0.040 - 0.054 * 0.034
    assert section.area == expected_area
    assert section.ix > 0
    assert section.iy > 0
    assert section.wx > 0
    assert section.wy > 0
    assert section.mass_per_m > 0


def test_square_tube_torsion_uses_closed_wall_shear_flow():
    # For a thin square tube J = t * a_m**3; unlike the polar area moment.
    section = parse_rhs("100x100x3")
    assert section.torsion_j == pytest.approx(0.003 * 0.097**3)
    assert section.torsion_j < section.ix + section.iy


@pytest.mark.parametrize("name", ["0x40x3", "60x40x0", "60x40x-3", "nanx40x3", "infx40x3"])
def test_section_rejects_nonphysical_dimensions(name):
    with pytest.raises(ValueError):
        parse_rhs(name)
