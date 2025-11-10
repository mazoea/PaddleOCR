# coding=utf-8
# See main file for licence
# pylint: disable=R0904

"""
    bbox wh implementation
"""


class bbox(object):
    """
        Bbox specified by x,y,w,h
    """

    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return "<bbox_wh> lt=%s rb=%s w=%s h=%s" % (
            [round(float(x), 1) for x in self.lt],
            [round(float(x), 1) for x in self.rb],
            round(float(self.d["w"]), 1),
            round(float(self.d["h"]), 1)
        )

    def __eq__(self, other):
        if hasattr(other, "d"):
            return self.d == other.d
        return False
        # if "x" in other.d and "w" in other.d:
        #     return self.d["x"] == other.d["x"] \
        #         and self.d["y"] == other.d["y"] \
        #         and self.d["w"] == other.d["w"] \
        #         and self.d["h"] == other.d["h"]
        # raise NotImplemented

    # ================================

    @property
    def x(self):
        """ left most x """
        return self.d["x"]

    @property
    def y(self):
        """ top most y """
        return self.d["y"]

    @property
    def width(self):
        return self.d["w"]

    @property
    def height(self):
        return self.d["h"]

    # left/bottom coordinate
    #

    @property
    def xl(self):
        """ x left """
        return self.d["x"]

    @property
    def xr(self):
        """ x right """
        return self.d["x"] + self.d["w"]

    @property
    def yt(self):
        """ y top """
        return self.d["y"]

    @property
    def yb(self):
        """ y bottom """
        return self.d["y"] + self.d["h"]

    def mid_x(self):
        """ X - coordinate of the middle word on the line. """
        return (self.d["x"] + self.xr) / 2

    def mid_y(self):
        """ Y- coordinate of the middle point on the line. """
        return (self.d["y"] + self.yb) / 2

    # points
    #

    @property
    def lt(self):
        return self.d["x"], self.d["y"]

    @property
    def rb(self):
        return self.d["x"] + self.d["w"], self.d["y"] + self.d["h"]

    @property
    def rt(self):
        return self.d["x"] + self.d["w"], self.d["y"]

    @property
    def lb(self):
        return self.d["x"], self.d["y"] + self.d["h"]

    @property
    def center(self):
        return self.mid_x(), self.mid_y()

    # helpers
    #

    def volume(self):
        return self.d["h"] * self.d["w"]

    def round(self):
        """
            Round
        """
        # we cannot just round all values including width
        # because rounding xl already changes xr (xr + width)
        new_xr = int(self.xr + 0.5)
        new_yb = int(self.yb + 0.5)
        self.d["x"] = int(self.d["x"] + 0.5)
        self.d["y"] = int(self.d["y"] + 0.5)
        self.d["w"] = new_xr - self.d["x"]
        self.d["h"] = new_yb - self.d["y"]
        return self

    def extend(self, diff_float):
        self.d["x"] -= diff_float
        self.d["y"] -= diff_float
        self.d["w"] += 2 * diff_float
        self.d["h"] += 2 * diff_float
        return self

    def scale(self, ratio_float):
        """
            Scale (multiply) every item with ratio.
        """
        for k, v in self.d.items():
            self.d[k] = ratio_float * v
        return self

    def relative_to(self, xy):
        """
            Update bbox relative to.
        """
        x, y = xy
        self.d["x"] -= x
        self.d["y"] -= y
        return self

    def copy(self):
        return bbox(self.d.copy())

    def list_2corner(self):
        return [self.d["x"], self.d["y"], self.xr, self.yb]

    def points_2corner(self):
        return [self.lt, self.rb]

    def points_4corner(self):
        return [self.lt, self.rt, self.rb, self.lb]

    def dict(self, res_type=int):
        d = self.d.copy()
        for k, v in d.items():
            d[k] = res_type(v)
        return d

    def offset_yt(self, val):
        newd = self.d.copy()
        newd["y"] += val
        newd["h"] -= val
        return bbox(newd)

    def offset_yb(self, val):
        newd = self.d.copy()
        newd["h"] += val
        return bbox(newd)

    @staticmethod
    def valid(d):
        """
            Return if it is in the correct form..
        """
        if "x" not in d:
            return False
        if "w" not in d:
            return False
        return True
