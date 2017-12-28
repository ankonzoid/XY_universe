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
    A1_x, A1_y = ptA1
    A2_x, A2_y = ptA2
    dx_A, dy_A = A2_x - A1_x, A2_y - A1_y
    # the second line is ptB1 + s*(ptB2 - ptB1)
    B1_x, B1_y = ptB1
    B2_x, B2_y = ptB2
    dx_B, dy_B = B2_x - B1_x, B2_y - B1_y
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
    DET = (-dx_A * dy_B + dy_A * dx_B)  # if DET small = parallel
    if np.abs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)  # lines are parallel if DET is too small
    DETinv = 1.0 / DET  # now, the determinant should be OK
    r = DETinv * (-dy_B * (B1_x - A1_x) + dx_B * (B1_y - A1_y))  # find the scalar amount along the "self" segment
    s = DETinv * (-dy_A * (B1_x - A1_x) + dx_A * (B1_y - A1_y))  # find the scalar amount along the input line
    xi = (A1_x + r * dx_A + B1_x + s * dx_B) / 2.0  # return the average of the two descriptions
    yi = (A1_y + r * dy_A + B1_y + s * dy_B) / 2.0  # return the average of the two descriptions
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
