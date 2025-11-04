# coding=utf-8
# See main file for licence
# pylint: disable=W0401,R0914

"""
    Angle
"""

import math


def degree2radian(deg):
    """ Degree to radian conversion. """
    return math.radians(deg)


def radian2degree(rad):
    """ Radian to degree conversion. """
    return math.degrees(rad)


def two_point_degree(xy1, xy2):
    """ Return angle in degrees. """
    (x1, y1) = xy1
    (x2, y2) = xy2
    deltax = x2 - x1
    deltay = y2 - y1

    angle_rad = math.atan2(deltay, deltax)
    return radian2degree(angle_rad)


def make_rotation_positive(rotation):
    """
        If the rotation is negative, make it positive.
    """
    angle_degree = radian2degree(rotation)
    # make angle it > 0
    if angle_degree < 0:
        angle_degree += 360
        angle_degree %= 360
    return degree2radian(angle_degree)
