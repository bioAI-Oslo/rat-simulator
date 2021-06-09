import numpy as np

from .methods import intersect


class LinearWall:
    def __init__(self, name, bias, slope, t=[0, 1]):
        """
        line-segment: u+tv
        """
        self.name = name
        self.bias = bias
        self.slope = slope
        self.t = t

        self.intersects = {}

    @property
    def end(self):
        return self.bias + self.t[-1] * self.slope

    @property
    def isborderwall(self):
        # defines the boundaries of the environment
        return "border_wall" in self.name

    @property
    def iscorner(self):
        """
        'Returns' two booleans in a tuple (e.g. (True, False) )
            that denote whether the bias or the end point intersects other walls
        """
        b, e = False, False
        for intersec in self.intersects.values():
            b = b or (self.bias == intersec).all()
            e = e or (self.end == intersec).all()
        """
        return (
            self.bias in self.intersects.values(),
            self.end in self.intersects.values(),
        )
        """
        return b, e

    def save_intersect(self, wall):
        """
        Save all intersections that self.wall is involved in
        """
        intersection, valid_intersect = intersect(
            self.bias, self.slope, wall.bias, wall.slope, self.t, wall.t
        )

        # intersection outside environment - i.e "the walls do not intersect"
        if not valid_intersect:
            return None

        # two walls should not cross, except at end points (of at least one wall)
        end_intersect = (
            (intersection == self.bias).all()
            or (intersection == self.end).all()
            or (intersection == wall.bias).all()
            or (intersection == wall.end).all()
        )

        assert end_intersect, "Walls should only intersect at end points, not mid wall!"

        # save wall intersection
        self.intersects[wall] = intersection
        wall.intersects[self] = intersection

    def normals(self):
        """
        2D (unit) normal vectors of wall (wrt. slope-vector)

        The normal vectors n1,n2 to a vector v=(vx,vy)=(v[0],v[1]) in R^2
        is the vectors n1 = (-vy,vx) = (-v[1],v[0]) and n2 = (vy,-vx) = (v[1],-v[0])
        More in depth-explanation and how to calculate 2d-normals:
        https://stackoverflow.com/questions/1243614/how-do-i-calculate-the-normal-vector-of-a-line-segment
        """
        v = self.slope
        n1 = np.array([-v[1], v[0]])
        n1 = n1 / np.sqrt(sum(n1 ** 2))  # unit normal
        return n1, -n1
