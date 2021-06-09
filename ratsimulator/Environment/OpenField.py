import numpy as np

from .ABCEnvironment import ABCEnvironment


class OpenField(ABCEnvironment):
    def __init__(self, low=(0, 0), high=(2.2, 2.2)):
        """
        high,low define OpenField sampling range
        """
        self.low, self.high = low, high

    def sample_uniform(self, ns=1):
        """
        Uniform sampling a 2d-rectangle is trivial with numpy
        """
        return np.random.uniform(self.low, self.high, size=(ns, 2))

    def avoid_walls(self, pos, hd, speed, turn):
        """No walls to avoid, return sampled speed and turn as is"""
        return speed, turn

    def plot_board(self, ax):
        """Environment does not have any walls to plot"""
        return None
