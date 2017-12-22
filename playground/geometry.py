

if __name__ == "__main__":

    from sympy import Point, Line
    l1 = Line(Point(0, 0), Point(0.8, 1))
    l2 = Line(Point(5, -1), Point(0, 3))
    A = l1.intersection(l2)
    print(A)
    print(A[0])
    print(A[0][0])
    print(A[0][1])
