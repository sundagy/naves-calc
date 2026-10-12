"""Guard published table values and prevent extrapolation in the scoped check."""
import pytest
from scripts.check_sp16_details import phi_e


def test_published_d3_coefficients_and_interpolation():
    # Independently transcribed points of SP16 D.3.
    assert phi_e(.5,.1) == pytest.approx(.967)
    assert phi_e(2,.25) == pytest.approx(.742)
    assert phi_e(4,.5) == pytest.approx(.394)
    assert phi_e(1.25,.375) == pytest.approx((.854+.778+.804+.716)/4)


def test_small_eccentricity_is_conservatively_clamped():
    assert phi_e(1,0) == phi_e(1,.1)
    assert phi_e(2,.001) == phi_e(2,.1)


@pytest.mark.parametrize('lam,m',[(.4,.1),(4.1,.1),(1,.51),(1,-.1)])
def test_outside_verified_table_is_not_extrapolated(lam,m):
    with pytest.raises(ValueError):
        phi_e(lam,m)
