import numpy as np
from scipy.spatial.distance import euclidean

from .ABCEnvironment import ABCEnvironment
from .Walls import LinearWall
from .methods import intersect, projection_rejection


class Rectangle(ABCEnvironment):
    def __init__(self, boxsize=(2.2, 2.2), soft_boundary=0.3):
        self.origo = np.array((0, 0))  # bottom left coordinate
        self.boxsize = np.array(boxsize)  # top right coordinate
        self.soft_boundary = soft_boundary

        # init walls
        self.walls = []
        self.add_wall(
            name="border_wall1",
            bias=self.origo,
            slope=np.array([self.origo[0], self.boxsize[1]]),
            t=[0, 1],
        )
        self.add_wall(
            name="border_wall2",
            bias=np.array([self.origo[0], self.boxsize[1]]),
            slope=np.array([self.boxsize[0], self.origo[1]]),
            t=[0, 1],
        )
        self.add_wall(
            name="border_wall3",
            bias=np.array([self.boxsize[0], self.origo[1]]),
            slope=np.array([self.origo[0], self.boxsize[1]]),
            t=[0, 1],
        )
        self.add_wall(
            name="border_wall4",
            bias=self.origo,
            slope=np.array([self.boxsize[0], self.origo[1]]),
            t=[0, 1],
        )

    def get_board(self, res=(32, 32)):
        # initialize board
        xx, yy = np.meshgrid(
            np.linspace(self.origo[0], self.boxsize[0], res[0]),
            np.linspace(self.origo[1], self.boxsize[1], res[1]),
        )
        return np.stack([xx, yy], axis=-1)

    def plot_board(self, ax):
        """
        Plots board walls and soft boundaries
        """
        for wall in self.walls:
            # plottable wall matrix
            W = np.array([wall.bias, wall.end]).T
            ax.plot(*W, "red")

            n1, n2 = wall.normals()
            n1, n2 = n1 * self.soft_boundary, n2 * self.soft_boundary

            # don't plot soft boundary outside of border walls
            if (not wall.isborderwall) or self.inside_environment(
                wall.bias + np.mean(wall.t) * wall.slope + n1
            ):
                SB1 = (W.T + n1).T  # soft boundary 1
                ax.plot(*SB1, "orange")
            if (not wall.isborderwall) or self.inside_environment(
                wall.bias + np.mean(wall.t) * wall.slope + n2
            ):
                SB2 = (W.T + n2).T  # soft boundary 2
                ax.plot(*SB2, "orange")

    def sample_uniform(self, ns=1):
        """
        Uniform sampling a 2d-rectangle is trivial with numpy
        """
        return np.random.uniform(self.origo, self.boxsize, size=(ns, 2))

    def add_wall(self, name, bias, slope, t=[0, 1]):
        """Add wall to walls."""
        new_wall = LinearWall(name, bias, slope, t)

        # save new wall intersects
        for wall in self.walls:
            wall.save_intersect(new_wall)

        self.walls.append(new_wall)

    def inside_environment(self, pos):
        """
        Check if agent is inside the defined environment
        """
        return (self.boxsize >= (pos - self.origo)).all() and (pos >= self.origo).all()

    def crash_point(self, pos, vel):
        """
        Treat current position and velocity, and walls as line-segments.
        We can then find where the agent will crash on each wall
        by solving a system of linear equations for each wall.
        """
        nearest_intersection = None
        for wall in self.walls:
            intersection, valid_intersect = intersect(
                pos, vel, wall.bias, wall.slope, [0, np.inf], wall.t
            )

            if valid_intersect:  # along animal trajectory and inside env
                if (nearest_intersection is None) or (
                    euclidean(pos, nearest_intersection) > euclidean(pos, intersection)
                ):
                    nearest_intersection = intersection

        return nearest_intersection, wall

    def wall_rejection(self, pos):
        """
        Walls reject agent when it comes too close.
        ed (escape direction) is the direction the agent is rejected towards.
        """
        ed = np.zeros(2)
        for wall in self.walls:
            proj, rej = projection_rejection(pos - wall.bias, wall.slope)

            # Projection-vector must have correct direction and be contained
            # in the line-segment
            direction = proj @ wall.slope
            if not ((direction >= 0) and (direction <= wall.slope @ wall.slope)):
                continue

            d = euclidean(self.origo, rej)
            ed += int(d <= self.soft_boundary) * (
                rej / d
            )  # unit-rejection / wall normal vector

        return ed

    def avoid_walls(self, pos, hd, speed, turn):
        # --- Regulate turn ---
        ed = self.wall_rejection(pos)

        # score next animal direction wrt. wall escape direction
        score_p = ed @ [np.cos(hd + turn), np.sin(hd + turn)]
        score_n = ed @ [np.cos(hd - turn), np.sin(hd - turn)]
        if score_n > score_p:
            turn = -turn  # turn away from wall

        # --- Regulate speed ---
        direction = np.array([np.cos(hd + turn), np.sin(hd + turn)])
        intersection, _ = self.crash_point(pos, direction)

        # speed is maximum half the distance to the crash point
        speed = min(speed, euclidean(pos, intersection) / 2)
        return speed, turn
