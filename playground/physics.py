import numpy as np

# Function for agent sensing the wall
def _dist_point_line(x, line):
    (p1, p2) = (line[0], line[1])
    d_perp = np.linalg.norm(np.cross(p2 - p1, p1 - x)) / np.linalg.norm(p2 - p1)
    return d_perp

if __name__ == "__main__":
    A = [-2, -1, 0, 1, 2]
    B = []
    x_transfer = 0
    while x_transfer in A: A.remove(x_transfer); B.append(x_transfer)
    print(A)
    print(B)