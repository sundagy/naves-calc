from src.sections.rhs import parse_rhs


def test_rhs_properties():
    section = parse_rhs("60x40x3")
    expected_area = 0.060 * 0.040 - 0.054 * 0.034
    assert section.area == expected_area
    assert section.ix > 0
    assert section.iy > 0
    assert section.wx > 0
    assert section.wy > 0
    assert section.mass_per_m > 0

