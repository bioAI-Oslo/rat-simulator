import numpy as np
from scipy.spatial.distance import euclidean


def batch_trajectory_generator(batch_size=64, seq_len=20, *args, **kwargs):
    """Mini-batch trajectory generator"""
    tgen = trajectory_generator(*args, **kwargs)
    mb_pos, mb_vel = np.zeros((batch_size, seq_len + 1, 2)), np.zeros(
        (batch_size, seq_len + 1, 2)
    )
    while True:
        for i in range(batch_size):
            pos, vel = next(tgen)[:2]
            mb_pos[i] = pos
            mb_vel[i] = vel
        yield mb_pos, mb_vel


def trajectory_generator(environment, seq_len=20, angle0=None, p0=None, **kwargs):
    # create agent
    agent = Agent(environment, **kwargs)
    while True:
        # re-initialize agent
        agent.reset(angle0, p0)
        # generate track
        for i in range(seq_len):
            agent.step()
        yield agent.positions, agent.velocities, agent.hds, agent.speeds, agent.turns, agent


class Agent:
    def __init__(
        self,
        environment,
        angle0=None,
        p0=None,
        dt=0.02,
        turn_angle=5.76 * 2,
        b=0.13 * 2 * np.pi,
        mu=0,
        boundary_mode="zenos",
        **kwargs,
    ):
        """
        default constants are the ones Sorscher used
        """
        self.environment = environment
        self.dt = dt
        self.turn_angle = turn_angle  # stdev rotation velocity (rads/sec)
        self.b = b  # forward velocity rayleigh dist scale (m/sec)
        self.mu = mu  # turn angle bias
        self.boundary_mode = boundary_mode  # boundary avoidance method

        self.reset(angle0, p0)

    def reset(self, angle0=None, p0=None):
        # N+1 len array histories (since we include start pos and hd)
        self.hds = (
            np.random.uniform(0, 2 * np.pi, size=1)
            if angle0 is None
            else np.array([angle0])
        )  # head direction history
        self.speeds = np.zeros(1)  # speed history
        self.turns = np.zeros(1)  # turn direction history
        self._velocities = np.zeros(
            (0, 2)
        )  # velocity history (also N+1, but only when called)
        self._positions = (
            self.environment.sample_uniform(1) if p0 is None else p0
        )  # position history

    def step(self, record_step=True):
        """
        Sample a velocity vector - indirectly through speed
        and angle, i.e. (s,phi). The angle is an offset to
        the angle at the previous time step.
        """
        new_speed = np.random.rayleigh(self.b) * self.dt
        new_turn = np.random.normal(self.mu, self.turn_angle) * self.dt

        new_speed, new_turn = self.avoid_walls(
            self.positions[-1], self.hds[-1], new_speed, new_turn
        )
        new_hd = np.mod(self.hds[-1] + new_turn, 2 * np.pi)

        if record_step:
            self.speeds = np.append(self.speeds, new_speed)
            self.hds = np.append(self.hds, new_hd)
            self.turns = np.append(self.turns, new_turn)

        return new_speed, new_hd

    def avoid_walls(self, *args, **kwargs):
        if self.boundary_mode == "zenos":
            return self.zenos_avoid_walls(*args, **kwargs)
        elif self.boundary_mode == "sorschers":
            return self.sorschers_avoid_walls(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.boundary_mode=} not implemented")

    def sorschers_avoid_walls(self, pos, hd, speed, turn):
        # find escape direction (wrt. box-clipped position. Since sorscher allows
        # agent to travel outside box - but regulates quick recovery).
        box_clipped_pos = np.clip(
            pos, self.environment.origo + 1e-8, self.environment.boxsize - 1e-8
        )
        ed = self.environment.wall_rejection(box_clipped_pos)

        # if escape direction is [0,0], then agent not in soft boundary
        # and no turn nor speed needs regulating
        if (ed != 0).any():
            next_hd_toward_wall = (ed @ [np.cos(hd + turn), np.sin(hd + turn)]) < 0
            if next_hd_toward_wall:
                # score next animal direction wrt. wall escape direction
                score_p = ed @ [
                    np.cos(hd + turn + np.pi / 2),
                    np.sin(hd + turn + np.pi / 2),
                ]
                score_n = ed @ [
                    np.cos(hd + turn - np.pi / 2),
                    np.sin(hd + turn - np.pi / 2),
                ]
                # --- Regulate turn, instanteneously move head +/- 90degrees ---
                turn += np.sign(score_p - score_n) * np.pi / 2
                # --- Regulate speed ---
                speed = speed / 4

        return speed, turn

    def zenos_avoid_walls(self, pos, hd, speed, turn):
        # --- Regulate turn ---
        ed = self.environment.wall_rejection(pos)
        # score next animal direction wrt. wall escape direction
        score_p = ed @ [np.cos(hd + turn), np.sin(hd + turn)]
        score_n = ed @ [np.cos(hd - turn), np.sin(hd - turn)]
        if score_n > score_p:
            turn = -turn  # turn away from wall

        # --- Regulate speed, Zeno style ---
        direction = np.array([np.cos(hd + turn), np.sin(hd + turn)])
        intersection, _ = self.environment.crash_point(pos, direction)
        # speed is maximum half the distance to the crash point
        speed = min(speed, euclidean(pos, intersection) / 2)
        return speed, turn

    @property
    def velocities(self):
        """
        Euclidean velocity history
        """
        idx0 = self._velocities.shape[0]
        if idx0 == self.speeds.shape[0]:
            return self._velocities

        direction = np.stack(
            [np.cos(self.hds[idx0:]), np.sin(self.hds[idx0:])], axis=-1
        )
        velocity = direction * self.speeds[idx0:][..., None]
        self._velocities = np.concatenate([self._velocities, velocity])
        return self._velocities

    @property
    def positions(self):
        """
        Path integration (Euclidean position) history
        """
        idx0 = self._positions.shape[0]
        if idx0 == self.speeds.shape[0]:
            return self._positions

        delta_p = np.cumsum(self.velocities[idx0:], axis=0)
        self._positions = np.concatenate(
            [self._positions, delta_p + self._positions[-1]]
        )
        return self._positions

    def plot_trajectory(self, ax, positions=None, ds=4):
        """plot animal path"""
        # fading color for early positions (wrt. time) of path positions
        plot_fancy_arrows = True if positions is None else False
        positions = self.positions if positions is None else positions
        n = positions.shape[0]
        c = np.zeros((n, 4))
        c[:, -1] = 1
        c[:, :-1] = 0.9 - np.linspace(0, 0.9, n)[:, None]

        # plot animal path
        ax.scatter(*positions.T, s=0.1, c=c)

        if plot_fancy_arrows:
            # add fancy velocity-arrows to trajectory plot
            i = 0
            for pos, vel in zip(positions[::ds], self.velocities[::ds]):
                ax.arrow(*pos, *vel, head_width=0.02, color=c[::ds][i])
                i += 1
