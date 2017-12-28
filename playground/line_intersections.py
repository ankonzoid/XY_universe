#
# intersections.py
#
# Python for finding line intersections
#   intended to be easily adaptable for line-segment intersections
#

import math
import numpy as np

def intersectLines(ptA1, ptA2, ptB1, ptB2):
    """ this returns the intersection of Line(ptA1,ptA2) and Line(ptB1,ptB2)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """
    DET_TOLERANCE = 0.00000001
    # the first line is ptA1 + r*(ptA2 - ptA1)
    x1, y1 = ptA1
    x2, y2 = ptA2
    dx1 = x2 - x1
    dy1 = y2 - y1
    # the second line is ptB1 + s*(ptB2 - ptB1)
    x, y = ptB1
    xB, yB = ptB2
    dx, dy = xB - x, yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)  # if DET small = parallel
    if np.abs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)
    DETinv = 1.0 / DET  # now, the determinant should be OK
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))  # find the scalar amount along the "self" segment
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))  # find the scalar amount along the input line
    xi = (x1 + r * dx1 + x + s * dx) / 2.0  # return the average of the two descriptions
    yi = (y1 + r * dy1 + y + s * dy) / 2.0  # return the average of the two descriptions
    return (xi, yi, 1, r, s)

def testIntersection(pt1, pt2, ptA, ptB):
    print("Line segment #1 runs from", pt1, "to", pt2)
    print("Line segment #2 runs from", ptA, "to", ptB)
    result = intersectLines(pt1, pt2, ptA, ptB)
    print("    Intersection result =", result)

if __name__ == "__main__":
    pt1, pt2 = [0, 0], [-1, -1]
    pt3, pt4 = [10, 0], [0, 10]
    testIntersection(pt1, pt2, pt3, pt4)
