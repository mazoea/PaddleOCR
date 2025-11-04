# coding=utf-8
# See main file for licence
# pylint: disable=W0401,R0914,C0200

"""
    Point
"""
import math


def distance(p1_tuple, p2_tuple):
    """
        Get distance of two points.
    """
    return math.sqrt((p1_tuple[0] - p2_tuple[0]) ** 2 +
                     (p1_tuple[1] - p2_tuple[1]) ** 2)


def left_corners(points):
    """
        Get the top ones, starting from top.
    """
    x_sort = sorted(points, key=lambda p: p[0])
    x_sort = x_sort[:2]
    y_sort = sorted(x_sort, key=lambda p: p[1])
    return y_sort


def right_corners(points):
    """
        Get the bottom ones, starting from top
    """
    x_sort = sorted(points, key=lambda p: p[0])
    x_sort = x_sort[2:]
    y_sort = sorted(x_sort, key=lambda p: p[1])
    return y_sort


def transpose(x, y, deskew_angle_radian):
    """
        Deskew one point.
    """

    def _tran_x(x1, y1):
        """ Transpose x coordinate. """
        return x1 * math.cos(deskew_angle_radian) - y1 * \
            math.sin(deskew_angle_radian)

    def _tran_y(x1, y1):
        """ Transpose y coordinate. """
        return x1 * math.sin(deskew_angle_radian) + y1 * \
            math.cos(deskew_angle_radian)

    return _tran_x(x, y), _tran_y(x, y)


def create_trans_matrix(rot_rad, deskew_rad, scale):
    """
        From rotation, deskew and scale create transformation matrix.
        Transformation matrix in this case is 3x3 matrix, which reflects transformation, which was done with
        original picture before OCR.
        Transformation matrix is created by composition from rotate and scale matrix
        (also other transformation can be represented by matrix, but they are not used in our case).
    """
    # -1 make rotation clockwise
    r = -1 * (rot_rad + deskew_rad)
    rot_matrix = [[math.cos(r), -(math.sin(r)), 0],
                  [math.sin(r), math.cos(r), 0],
                  [0, 0, 1]]
    scale_matrix = [[scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1]]
    result_matrix = [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]

    # we do not want use numpy, so write it
    for i in range(len(rot_matrix)):
        for j in range(len(scale_matrix[0])):
            for k in range(len(scale_matrix)):
                result_matrix[i][j] += rot_matrix[i][k] * scale_matrix[k][j]

    return result_matrix
