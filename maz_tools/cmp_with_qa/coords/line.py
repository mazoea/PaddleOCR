# coding=utf-8
# See main file for licence
# pylint: disable=W0401,R0914

"""
    Line
"""


# pylint: disable=R0911
def overlap(pl1, pr1, pl2, pr2):
    """
        Checks if two lines overlap.

        Returns

          Returns the percentage of the "better match"
          which overlap i.e., the bigger number the
          more they overlap.
    """
    assert pl1 <= pr1
    assert pl2 <= pr2

    if pl1 == pl2 and pr1 == pr2:
        return 1.0

    # widths of lines (min 1 because we divide)
    l1_w = float(max(1., pr1 - pl1))
    l2_w = float(max(1., pr2 - pl2))

    # for the first "line", check if one
    # of its points is inside the other
    # - use sharp <, we want overlap anyway
    #
    pl1_inside = pl2 < pl1 < pr2
    pr1_inside = pl2 < pr1 < pr2
    pl2_inside = pl1 < pl2 < pr1
    pr2_inside = pl1 < pr2 < pr1

    if pl1_inside:
        if pr1_inside:
            return 1.0  # full overlap
        till = pr2
        return max(float(till - pl1) / l1_w, float(till - pl1) / l2_w)

    # not full overlap
    assert not (pl1_inside and pr1_inside)
    if pr1_inside:
        fromm = pl2
        return max(float(pr1 - fromm) / l1_w,
                   float(pr1 - fromm) / l2_w)
    if pl2_inside:
        if pr2_inside:
            return 1.0  # full overlap
        till = pr1
        return max(float(till - pl2) / l1_w, float(till - pl2) / l2_w)

    # not full overlap
    assert not (pl2_inside and pr2_inside)
    if pr2_inside:
        fromm = pl1
        return max(float(pr2 - fromm) / l1_w,
                   float(pr2 - fromm) / l2_w)
    return 0.0
