# coding=utf-8
# See main file for licence
# pylint: disable=W0401,R0914,W0612,R0913,E0202

"""
    Bbox
"""
import math

import json
try:
    import simplejson  # type: ignore
except Exception:
    simplejson = json  # type: ignore

json_library = json
# according to tests loading is faster with simplejson
#
json_library_load = simplejson
json_library_dump = json

from . import angle
from . import line
from . import point

from . import bbox_wh
default_bbox = bbox_wh.bbox


# ================================
# creation
# ================================

def create_from_points(xlt, ylt, xrb, yrb):
    return default_bbox({
        "x": xlt,
        "y": ylt,
        "w": xrb - xlt,
        "h": yrb - ylt
    })


def create(d):
    """
        Create bbox from dictionary.
    """
    from . import bbox_x1
    if bbox_wh.bbox.valid(d):
        return bbox_wh.bbox(d)
    if bbox_x1.bbox.valid(d):
        return bbox_x1.bbox(d)
    raise NotImplementedError("not implemented for four corner")


class json_encoder(json_library.JSONEncoder):
    """
        Convert document object instances (e.g., bboxes)
        into basic json types.
    """

    def default(self, o):
        if isinstance(o, default_bbox):
            d = o.dict()
            for k, v in d.items():
                d[k] = int(v)
            return d
        # all other types should be primitive
        return json_library.JSONEncoder.default(self, o)


def json_decoder_hook(parsed_dict, object_hook_handler=None):
    """
        Convert string to bbox
    """
    # bbox
    if default_bbox.valid(parsed_dict):
        return default_bbox(parsed_dict)
    # user specified
    if object_hook_handler is not None:
        return object_hook_handler(parsed_dict)
    # default
    return parsed_dict


# ================================
# usage
# ================================

def volume(b):
    """
        Return area (w*h) of the bbox.
    """
    return b.width * b.height


def further_from(b1, b2, b_anchor):
    """
        Returns -1 if b1 is closer to b_anchor.
        Returns 1 if b2 is closer to b_anchor.

        Can be used as comparator.
    """
    x1, y1 = b1.center
    x2, y2 = b2.center
    xc, yc = b_anchor.center
    d1 = point.distance((x1, y1), (xc, yc))
    d2 = point.distance((x2, y2), (xc, yc))

    if d1 < d2:
        return -1
    if d1 > d2:
        return 1
    return 0


# ================================
# overlap
# ================================

def contains(b, x, y):
    return b.x <= x <= b.xr and b.y <= y <= b.yb


def x_overlap(b1, b2, extend_to_left=0, extend_to_right=0):
    """ Get x overlap of two bboxes """
    if 0 == extend_to_left and 0 == extend_to_right:
        return line.overlap(b1.x, b1.xr, b2.x, b2.xr)
    # b1 and then b2
    if b1.xr < b2.xr:
        r1, r2 = b1, b2
        extend_to_left = 0
    else:
        r1, r2 = b2, b1
        extend_to_right = 0
    return r1.xr + extend_to_left > r2.xl - extend_to_right


def y_overlap(b1, b2, y_will=0):
    """
        Return if y axis overlaps
    """
    r1, r2 = (b1, b2) if b1.yt < b2.yt else (b2, b1)
    y_will = y_will or (r1.yb - r1.yt) / 5
    # return if bottom ( - small fragment) is above top
    return (r1.yb - y_will) > r2.yt


def overlap(b1, b2, ratio=True):
    """
        Return the overlap area of two rectangles.
    """
    volume_b1 = volume(b1)
    volume_b2 = volume(b2)
    if volume_b1 == 0 or volume_b2 == 0:
        return 0

    overlap_x = max(0, min(b1.xr, b2.xr) - max(b1.xl, b2.xl))
    overlap_y = max(0, min(b1.yb, b2.yb) - max(b1.yt, b2.yt))
    overlap_vol = overlap_x * overlap_y
    bigger_vol = max(volume_b1, volume_b2)
    if ratio:
        return round(float(overlap_vol) / float(bigger_vol), 2)
    return overlap_vol


def overlap_with(b1, b2, ratio=True):
    """
        Return the overlap area of two rectangles in respect
        to the second one.
    """
    volume_b1 = volume(b1)
    volume_b2 = volume(b2)
    if volume_b1 == 0 or volume_b2 == 0:
        return 0

    overlap_x = max(0, min(b1.xr, b2.xr) - max(b1.xl, b2.xl))
    overlap_y = max(0, min(b1.yb, b2.yb) - max(b1.yt, b2.yt))
    overlap_vol = overlap_x * overlap_y
    if ratio:
        return round(float(overlap_vol) / float(volume_b2), 2)
    return overlap_vol


def overlap_min(b1, b2):
    """
        Return the overlap area of two rectangles.
    """
    volume_b1 = volume(b1)
    volume_b2 = volume(b2)
    if volume_b1 == 0 or volume_b2 == 0:
        return 0

    overlap_x = max(0, min(b1.xr, b2.xr) - max(b1.xl, b2.xl))
    overlap_y = max(0, min(b1.yb, b2.yb) - max(b1.yt, b2.yt))
    overlap_vol = overlap_x * overlap_y
    smaller_vol = min(volume_b1, volume_b2)
    if smaller_vol == 0:
        return 0
    return round(float(overlap_vol) / float(smaller_vol), 2)


def union(*arr):
    """
        Get union of two areas.
    """
    xl = min(a.xl for a in arr)
    yt = min(a.yt for a in arr)
    xr = max(a.xr for a in arr)
    yb = max(a.yb for a in arr)
    return default_bbox({
        "x": xl,
        "y": yt,
        "w": xr - xl,
        "h": yb - yt
    })


# ================================
# deskew/transpose
# ================================

def __bigger_than_90(radian_angle):
    return 89.5 <= abs(angle.radian2degree(radian_angle))


def deskew(b,
           deskew_angle_radian,
           document_b,
           center_pair,
           recreate=True,
           special_90=True):
    """
        Deskew a bbox.
    """
    center_x, center_y = center_pair

    # first rotate
    rotated_b = b
    # special cases when rotated with 90*x
    # because we use different rotation algorithms in image_to_text
    # which do not change coords so we have to have one 0, 0 (left topmost)
    if special_90 and recreate and __bigger_than_90(deskew_angle_radian):
        # if > 360
        deskew_angle_radian = angle.make_rotation_positive(deskew_angle_radian)
        while __bigger_than_90(deskew_angle_radian):
            # should be rotated to right?
            if 0 < angle.radian2degree(deskew_angle_radian):
                rotated_b, document_b = transpose_90_right(
                    rotated_b, (center_x, center_y), document_b
                )
                deskew_angle_radian -= angle.degree2radian(90)
            else:
                rotated_b, document_b = transpose_90_left(
                    rotated_b, (center_x, center_y), document_b
                )
                deskew_angle_radian += angle.degree2radian(90)
            # rotate around new center
            center_x, center_y = document_b.center

    # transpose
    transposed_b = transpose(
        rotated_b, deskew_angle_radian, (center_x, center_y)
    )

    # if there is a big rotation, x1 can by > x2 etc
    if recreate:
        transposed_b = recreate_transposed(transposed_b)

    return transposed_b


def transpose(b, deskew_angle_radian, c_arr):
    """
        Transpose bbox around center specified by cx_, cy.
    """
    c_x, c_y = c_arr
    if is_fourcorner(b):
        raise NotImplementedError("not implemented for four corner")

    # make bbox center the 0, 0 in x,y axis
    # with "normal" 4 quadrants in opposite to
    # what we use in bbox
    # euc - euclidean
    euc_xl = b.x - c_x
    euc_yt = c_y - b.yt
    euc_xr = b.xr - c_x
    euc_yb = c_y - b.yb

    # transpose
    trans_xlt, trans_ylt = point.transpose(
        euc_xl, euc_yt, deskew_angle_radian)
    trans_xrb, trans_yrb = point.transpose(
        euc_xr, euc_yb, deskew_angle_radian)
    b_trans = create_from_points(
        trans_xlt + c_x,
        c_y - trans_ylt,
        trans_xrb + c_x,
        c_y - trans_yrb
    )
    return b_trans


def transpose_90_left(b, c_arr, document_bbox):
    """ Turn 90 left. """
    c_x, c_y = c_arr
    return transpose_90(
        b, (c_x, c_y), document_bbox, angle.degree2radian(-90))


def transpose_90_right(b, c_arr, document_bbox):
    """ Turn 90 right. """
    c_x, c_y = c_arr
    return transpose_90(
        b, (c_x, c_y), document_bbox, angle.degree2radian(90))


def transpose_90(b, c_arr, document_bbox, radian_angle):
    """
        Transpose specifically around 90 degrees, the coordinates will be always
        inside the new mediabox because they will be relative to left top corner of
        the rotated mediabox
    """
    c_x, c_y = c_arr
    # 1st transpose the mediabox and find left top corner
    rotated_document_bbox = transpose(
        document_bbox, radian_angle, document_bbox.center)
    rotated_document_bbox = recreate_transposed(rotated_document_bbox)
    lt_corner, _1 = point.left_corners(rotated_document_bbox.points_4corner())
    rotated_document_bbox.relative_to(lt_corner)

    rotated_bbox = transpose(b, radian_angle, (c_x, c_y))
    rotated_bbox_final = recreate_transposed(rotated_bbox)
    rotated_bbox_final.relative_to(lt_corner)
    return rotated_bbox_final, rotated_document_bbox


def recreate_transposed(bbox):
    """
        If you transpose a bbox, the semantic can change (xlt does
        not need to be xlt anymore).
    """
    if is_fourcorner(bbox):
        raise NotImplementedError("not implemented for four corner")

    x = min(bbox.x, bbox.xr)
    y = min(bbox.y, bbox.yb)
    return default_bbox({
        "x": x,
        "y": y,
        "w": max(bbox.x, bbox.xr) - x,
        "h": max(bbox.y, bbox.yb) - y
    })


# ===========================
#
# ===========================

def is_fourcorner(bbox_or_d):
    """
        Return if it is in 2 corner mode.
    """
    bbox = bbox_or_d if isinstance(bbox_or_d, dict) else \
        bbox_or_d.d
    ret1 = (8 == len(bbox.keys()))
    ret2 = True
    for k in ("xlt", "ylt", "xrt", "yrt", "xrb", "yrb", "xlb", "ylb"):
        if k not in bbox:
            ret2 = False
            break
    return ret1 and ret2


# ===========================

def deskew_bboxes(deskew_deg: float, around_bb, arr: list) -> list:
    """
        Deskew bboxes.
    """
    if deskew_deg == 0.:
        return arr

    return [deskew(bb, -angle.degree2radian(deskew_deg), around_bb, center_pair=around_bb.center) for bb in arr]


def bbox_to_int(bbox_in: bbox_wh.bbox, img_width: int, img_height: int) -> bbox_wh.bbox:
    """
    Convert bbox to int. Checks for overflowing coordinates and keeps them inside the image.
    Returns: bbox
    """

    x = math.floor(bbox_in.d['x'])
    y = math.floor(bbox_in.d['y'])
    w = math.ceil(bbox_in.d['w'])
    h = math.ceil(bbox_in.d['h'])

    x_norm = max(min(x, img_width - 1), 0)
    y_norm = max(min(y, img_height - 1), 0)

    # make sure the rounded values are inside the image
    bbox_rectified = {
        "x": x_norm,
        "y": y_norm,
        "w": max(min(w, img_width - x_norm), 0),
        "h": max(min(h, img_height - y_norm), 0),
    }
    return default_bbox(bbox_rectified)
