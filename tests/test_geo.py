import math

import numpy as np
import pytest

from pcas.data.geo import FEET_TO_METERS, KBTP, LocalFrame, geodetic_to_ecef


def test_origin_maps_to_zero():
    frame = LocalFrame(lat_deg=40.7769, lon_deg=-79.9498, alt_m=380.0)
    x, y, z = frame.to_local(np.array([40.7769]), np.array([-79.9498]), np.array([380.0]))
    assert abs(float(x[0])) < 1e-6
    assert abs(float(y[0])) < 1e-6
    assert abs(float(z[0])) < 1e-6


def test_altitude_difference_becomes_up():
    frame = LocalFrame(lat_deg=40.0, lon_deg=-80.0, alt_m=0.0)
    _, _, z = frame.to_local(np.array([40.0]), np.array([-80.0]), np.array([1000.0]))
    assert float(z[0]) == pytest.approx(1000.0, abs=0.01)


def test_one_degree_of_latitude_is_about_111_km():
    # Heading 0 means the runway points north, so along-runway x IS north here.
    frame = LocalFrame(lat_deg=40.0, lon_deg=-80.0, runway_heading_deg=0.0)
    along, _, _ = frame.to_local(np.array([41.0]), np.array([-80.0]), np.array([0.0]))
    # ~111.0 km per degree of latitude at mid-latitudes; a few hundred m of tangent-plane
    # error over 1 degree is expected and harmless at airport scale.
    assert 110_000 < float(along[0]) < 112_000


def test_longitude_shrinks_with_latitude():
    # A degree of longitude is ~85 km at Pittsburgh and ~111 km at the equator. This is
    # exactly why models get metres, not degrees. Heading 90 puts east on x.
    at_equator = LocalFrame(lat_deg=0.0, lon_deg=0.0, runway_heading_deg=90.0)
    at_kbtp = LocalFrame(lat_deg=40.7769, lon_deg=-79.9498, runway_heading_deg=90.0)

    east_eq, _, _ = at_equator.to_local(np.array([0.0]), np.array([1.0]), np.array([0.0]))
    east_pit, _, _ = at_kbtp.to_local(np.array([40.7769]), np.array([-78.9498]), np.array([0.0]))

    assert float(east_eq[0]) > 111_000
    assert 84_000 < float(east_pit[0]) < 85_500


def test_axes_are_runway_relative_not_east_north():
    # Heading 0: x is along the runway (north) and y is to its left (west). A due-east
    # point therefore lands on -y, not +x.
    frame = LocalFrame(lat_deg=40.0, lon_deg=-80.0, runway_heading_deg=0.0)
    along, left, _ = frame.to_local(np.array([40.0, 40.01]), np.array([-79.99, -80.0]), np.zeros(2))

    assert abs(float(along[0])) < 1.0 and float(left[0]) < -800  # due east -> left is west
    assert float(along[1]) > 800 and abs(float(left[1])) < 1.0  # due north -> along


def test_rotation_puts_runway_direction_on_x():
    # With a 90-degree (due east) runway heading, a point due east is straight down the
    # runway: +x and ~zero y.
    frame = LocalFrame(lat_deg=40.0, lon_deg=-80.0, runway_heading_deg=90.0)
    along, cross, _ = frame.to_local(np.array([40.0]), np.array([-79.99]), np.array([0.0]))
    assert float(along[0]) > 800
    assert abs(float(cross[0])) < 1.0


def test_rotation_is_right_handed_y_points_left():
    # Runway heading due east, so "left of the runway" is north.
    frame = LocalFrame(lat_deg=40.0, lon_deg=-80.0, runway_heading_deg=90.0)
    _, cross, _ = frame.to_local(np.array([40.01]), np.array([-80.0]), np.array([0.0]))
    assert float(cross[0]) > 0


def test_rotation_preserves_distance():
    lat, lon = np.array([40.05]), np.array([-79.95])
    plain = LocalFrame(lat_deg=40.0, lon_deg=-80.0, runway_heading_deg=0.0)
    turned = LocalFrame(lat_deg=40.0, lon_deg=-80.0, runway_heading_deg=80.0)

    a = plain.to_local(lat, lon, np.zeros(1))
    b = turned.to_local(lat, lon, np.zeros(1))
    assert math.hypot(float(a[0][0]), float(a[1][0])) == pytest.approx(
        math.hypot(float(b[0][0]), float(b[1][0])), rel=1e-9
    )


def test_ecef_matches_known_equator_value():
    x, y, z = geodetic_to_ecef(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert float(x[0]) == pytest.approx(6378137.0, abs=0.001)
    assert abs(float(y[0])) < 1e-6
    assert abs(float(z[0])) < 1e-6


def test_kbtp_frame_is_configured_for_the_trajair_site():
    assert KBTP.runway_heading_deg == 80.0
    assert 40.7 < KBTP.lat_deg < 40.8
    assert FEET_TO_METERS == 0.3048
