# coding=utf-8
# See main file for licence
# pylint: disable=W0401,R0914,W0612,W0613

"""
  Simple plane full of lines.
"""

import sys
from . import line


def _diff_in_range(x1, x2, diff_acceptable):
    """ Returns whether two values are near according to diff_acceptable. """
    return abs(x1 - x2) <= diff_acceptable


class plane(object):
    """
        2 dimensional plane representation.

        .. note::

          The line difference is used in two places, first when you **add**
          fragments which means big acceptable diff will merge multiple lines into one,
          and during searching.

          Create a generic way of getting objects from 2d plane
          using ftors.

          Every ftor must implement three methods:
            * itery - returning NO 0, YES 1, NO_MORE  -1
            * iterx - storing results
            * results
      """
    NOT_ACCEPTED, ACCEPTED, STOP = list(range(3))
    LINE_DIFF = 15

    def __init__(self, key_bbox=None, line_height=None):
        self._values = []
        self._y_diff = line_height or plane.LINE_DIFF
        self._iter_ftor_class = matcher_in_rect
        self._key_bbox = key_bbox or (lambda x: x)

    def set_line_height(self, line_height):
        """
            Should be called after we added all of the fragments, so
            the acceptable line difference can be changed.
        """
        self._y_diff = line_height

    def add_line(self, y, iterable_objs):
        """
            Add array of objects represented by iterable_objs
            with y position.

            .. note:: The objects in the iterable must contain bbox.
        """
        pos, found = self.at_y(y)
        if not found:
            # (y, y_min, y_max) | arr
            self._values.insert(pos, {
                "y": y,
                "y_max": sys.maxsize,
                "y_min": -1,
                "items": []
            })
        to_put = self._values[pos]
        ys = []
        for o in iterable_objs:
            # get bbox not obj itself (if different)
            o_bbox = self._key_bbox(o)
            ys.append(o_bbox.yb)
            ys.append(o_bbox.yt)
            self._add_x(to_put["items"], o_bbox.xl, o)
        if 0 < len(ys):
            to_put["y_min"] = min(ys)
            to_put["y_max"] = max(ys)

    def at_y(self, y):
        """
            Return position of the line around y position or -1 if not found.

            .. note:

                Uses either `plane.LINE_DIFF` or argument supplied in ctor
                as the maximum difference  between the coordinates.
        """
        for i, line_d in enumerate(self._values):
            if _diff_in_range(line_d["y"], y, self._y_diff):
                return i, True
            if line_d["y"] > y:
                return i, False
        return len(self._values), False

    def at(self, in_rect_bbox, min_overlap=None):
        """
            Find object at position. The class representing the functor
            can be set.
        """
        # check whether the rect is ok
        iterator = self._iter_ftor_class(
            in_rect_bbox, self._y_diff, min_overlap)

        # give all lines
        for pos, line_d in enumerate(self._values):
            # does not make sense to check empty line
            if 0 == len(line_d["items"]):
                continue
            cont_result = iterator.itery(
                line_d["y"], line_d["y_min"], line_d["y_max"]
            )
            if plane.STOP == cont_result:
                break
            if plane.ACCEPTED == cont_result:
                # give all words
                for x_pos_line, obj in line_d["items"]:
                    cont_result = iterator.iterx(
                        x_pos_line, obj, self._key_bbox(obj)
                    )
                    if plane.STOP == cont_result:
                        break
        return iterator.result()

    def iteritems(self):
        for line_d in self._values:
            for i, (_1, word) in enumerate(line_d["items"]):
                yield word, i, line_d

    def lines(self):
        return self._values

    def items(self):
        arr = []
        for line_d in self._values:
            for i, (_1, word) in enumerate(line_d["items"]):
                arr.append((word, i, line_d))
        return arr

    def _at_x(self, line_arr, x):
        """
            Returns the position where we stopped looking and indicator whether
            we found it.

            E.g., (2, True)
        """
        for i, (x_there, _) in enumerate(line_arr):
            if _diff_in_range(x_there, x, self._y_diff):
                return i, True
            if x_there > x:
                return i, False
        return len(line_arr), False

    def _add_x(self, to_put, x, obj_inst):
        """
            Add `obj_inst` with x left position to to_put according to first value of the pairs
            which are stored there.
        """
        pos, _ = self._at_x(to_put, x)
        to_put.insert(pos, (x, obj_inst))


class matcher_in_rect(object):
    """
        Ftor implementation with
        rectangle overlapping that overlap by at least
        `MIN_OVERLAP`% / 100.

        `y` search is more loose
    """
    # test all lines because of te#3 where a word can be
    # outside of the line bbox
    MIN_OVERLAP_TO_CONSIDER = 0.0
    MIN_OVERLAP = 0.5
    # we use it in subtraction
    INT_MIN = -(sys.maxsize / 10)

    def __init__(self, bbox, _, min_overlap=None):
        """ Simple ctor. """
        self._search_bbox = bbox
        self._last_overlapped_y_max = matcher_in_rect.INT_MIN
        self._result_arr = []
        self._min_overlap = min_overlap if min_overlap is not None else self.MIN_OVERLAP

    def itery(self, _1, y_min, y_max):
        """
            Decides which y lines are interesting.
        """
        line_overlap = line.overlap(
            self._search_bbox.yt, self._search_bbox.yb, y_min, y_max
        )
        if line_overlap > self.MIN_OVERLAP_TO_CONSIDER:
            self._last_overlapped_y_max = y_max
            return plane.ACCEPTED

        if y_min > self._search_bbox.yb:
            # see te#3
            if (y_max - self._last_overlapped_y_max) < (y_max - y_min):
                self._last_overlapped_y_max = matcher_in_rect.INT_MIN
                return plane.ACCEPTED
            return plane.STOP

        return plane.NOT_ACCEPTED

    def iterx(self, _, o, b):
        """
            Stores the matching ones.
        """
        overlap_x = self._min_overlap < line.overlap(
            self._search_bbox.xl, self._search_bbox.xr, b.xl, b.xr
        )
        overlap = overlap_x and self._min_overlap < line.overlap(
            self._search_bbox.yt, self._search_bbox.yb, b.yt, b.yb
        )

        if overlap:
            self._result_arr.append(o)
        elif b.xl > self._search_bbox.xr:
            return plane.STOP
        return plane.ACCEPTED

    def result(self):
        """ Return collected objects. """
        return self._result_arr


class matcher_in_rect_simple(object):
    """
        Ftor simple implementation based on middle y and
        left x.
    """

    def __init__(self, b, y_diff):
        """ Simple ctor. """
        self._search_bbox = b
        self._result_arr = []
        self._y_diff = y_diff

    def itery(self, y, _1, _2):
        """
            Decides which y lines are interesting.
        """
        if _diff_in_range(y, self._search_bbox.mid_y, self._y_diff):
            return plane.ACCEPTED
        if y > self._search_bbox.mid_y:
            return plane.STOP
        return plane.NOT_ACCEPTED

    def iterx(self, x, obj):
        """
            Stores the matching ones.
        """
        if _diff_in_range(x, self._search_bbox.xl, self._y_diff):
            self._result_arr.append(obj)
        elif x > self._search_bbox.xl:
            return plane.STOP
        return plane.ACCEPTED

    def result(self):
        """ Return collected objects. """
        return self._result_arr
