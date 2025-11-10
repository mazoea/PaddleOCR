# coding=utf-8
# See main file for licence
# pylint: disable=W0401,W0702,R0914,F0401
"""
    Utils for ia module.
"""
from collections import defaultdict


class bboxes_in_blocks(object):
    """
        Indexed version of bbox for faster access.

        Supports distinguishing used entries.
    """

    def __init__(self, objs, block_size, only_corners=False):
        """
            Index all the bbox for fast searching
            the indexing is based on any corner of the bbox
            inside a block of given block_size
        """
        self._block_size = block_size
        self._key_bbox = bboxes_in_blocks._get_bbox
        self._index = defaultdict(list)
        self._only_corners = only_corners
        map(self.add, objs)
        self._used_keys = set()

    @property
    def block_size(self):
        return self._block_size

    def add(self, obj, only_corners=None):
        """
            Add one bbox to index.
        """
        if only_corners is None:
            only_corners = self._only_corners

        bbox = self._key_bbox(obj)
        left = bbox.xl / self._block_size
        top = bbox.yt / self._block_size
        right = bbox.xr / self._block_size
        bottom = bbox.yb / self._block_size

        if only_corners:
            self._index[(left, top)].append(obj)
            self._index[(right, top)].append(obj)
            self._index[(left, bottom)].append(obj)
            self._index[(right, bottom)].append(obj)
        else:
            for x in range(left, right + 1):
                for y in range(top, bottom + 1):
                    self._index[(x, y)].append(obj)

    def get(self, x, y):
        """
            Return boxes contains in the given key box.

            :returns: [objects]
        """
        self._used_keys.add((x, y))
        return self._index.get((x, y))

    def get_by_coords(self, x, y):
        """
            Return boxes contains in the given key box.

            :returns: [objects]
        """
        k = (x / self.block_size, y / self.block_size)
        self._used_keys.add(k)
        return self._index.get(k)

    def values(self):
        """ Get values. """
        return list(self._index.values())

    def dict(self):
        """ Get index dictionary. """
        return self._index

    def leftovers(self):
        ret = [x for x in list(self._index.keys()) if x not in self._used_keys]
        return ret

    def values_count(self):
        return sum(len(x) for x in list(self.values()))

    def iteritems(self):
        for k, v in self._index.items():
            yield k, v

    @staticmethod
    def _get_bbox(d):
        if isinstance(d, dict):
            return d.get("area", d["bbox"])
        return d.bbox
