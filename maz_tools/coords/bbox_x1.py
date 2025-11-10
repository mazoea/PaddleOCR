# coding=utf-8
# pylint: disable=R0904

"""
    bbox x1,y2,x2,y2 implementation
"""


class bbox(object):
    """
        Bbox specified by x1,y1,x2,y2
    """

    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return "<bbox_x1> lt=%s rb=%s w=%s h=%s" % (
            [round(x, 1) for x in self.lt],
            [round(x, 1) for x in self.rb],
            round(self.width, 1),
            round(self.height, 1)
        )

    def __eq__(self, other):
        if hasattr(other, "x") and hasattr(other, "width"):
            return self.x == other.x \
                and self.y == other.y \
                and self.width == other.width \
                and self.height == other.height
        return NotImplemented

    # ================================

    @property
    def x(self):
        """ left most x """
        return self.d["x1"]

    @property
    def y(self):
        """ top most y """
        return self.d["y1"]

    @property
    def width(self):
        return self.d["x2"] - self.x

    @property
    def height(self):
        return self.d["y2"] - self.y

    # left/bottom coordinate
    #

    @property
    def xl(self):
        """ x left """
        return self.x

    @property
    def xr(self):
        """ x right """
        return self.x + self.width

    @property
    def yt(self):
        """ y top """
        return self.y

    @property
    def yb(self):
        """ y bottom """
        return self.y + self.height

    def mid_x(self):
        """ X - coordinate of the middle word on the line. """
        return (self.x + self.xr) / 2

    def mid_y(self):
        """ Y- coordinate of the middle point on the line. """
        return (self.y + self.yb) / 2

    # points
    #

    @property
    def lt(self):
        return self.x, self.y

    @property
    def rb(self):
        return self.x + self.width, self.y + self.height

    @property
    def rt(self):
        return self.x + self.width, self.y

    @property
    def lb(self):
        return self.x, self.y + self.height

    @property
    def center(self):
        return self.mid_x(), self.mid_y()

    # helpers
    #

    def round(self):
        """
            Round
        """
        for k, v in self.d.items():
            self.d[k] = int(v)
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
        (x, y) = xy
        self.d["x1"] -= x
        self.d["y1"] -= y
        self.d["x2"] -= x
        self.d["y2"] -= y
        return self

    def copy(self):
        return bbox(self.d.copy())

    def list_2corner(self):
        return [self.x, self.y, self.xr, self.yb]

    def points_2corner(self):
        return [self.lt, self.rb]

    def points_4corner(self):
        return [self.lt, self.rt, self.rb, self.lb]

    def dict(self, res_type=int):
        d = self.d.copy()
        for k, v in d.items():
            d[k] = res_type(v)
        return d

    @staticmethod
    def valid(d):
        """
            Return if it is in the correct form..
        """
        expected = ("x1", "y1", "x2", "y2")
        if len(expected) != len(list(d.keys())):
            return False
        for k in expected:
            if k not in d:
                return False
        return True
