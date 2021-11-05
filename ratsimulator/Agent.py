import numpy as np
import sys
#sys.path.append("../") if "../" not in sys.path else None # avoid adding multiple relave paths to sys.path
#from ratsimulator.Environment import RectanglewObjects


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
        vfr = (np.pi/6, 0.1*np.pi),
        **kwargs
    ):
        """
        default constants are the ones Sorscher used
        """
        self.environment = environment
        self.dt = dt
        self.turn_angle = turn_angle  # stdev rotation velocity (rads/sec)
        self.b = b  # forward velocity rayleigh dist scale (m/sec)
        self.mu = mu  # turn angle bias
        self.vfr = vfr #tuple of visual field angular range and radial range
        self.reset(angle0, p0)
        # TODO! Change condition below to check if environment is with objects
        if True: #isinstance(environment, RectanglewObjects):
            self.envm_objects = self.fetch_objects()
            self.objects_observed = self.check_for_objects(self.envm_objects) #[] #list of objs observed at given positions
            self.observe = True
        else:
            self.observe = False
    

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

        # TODO! Create a history of objects seen by rat

    def step(self, record_step=True):
        """
        Sample a velocity vector - indirectly through speed
        and angle, i.e. (s,phi). The angle is an offset to
        the angle at the previous time step.
        """
        new_speed = np.random.rayleigh(self.b) * self.dt
        new_turn = np.random.normal(self.mu, self.turn_angle) * self.dt

        new_speed, new_turn = self.environment.avoid_walls(
            self.positions[-1], self.hds[-1], new_speed, new_turn
        )
        # Check to see if rat sees any objects with current pose (position and orientation)
        if self.observe:  
            self.objects_observed = np.append(self.objects_observed, self.check_for_objects(self.envm_objects), axis = 0)
            # objects_id_seen = self.check_for_objects(self.envm_objects)


            # if len(objects_id_seen) != 0: #remove this condition
            #     self.objects_observed.append((self.positions[-1], objects_id_seen)) 

        new_hd = np.mod(self.hds[-1] + new_turn, 2 * np.pi)

        if record_step:
            self.speeds = np.append(self.speeds, new_speed)
            self.hds = np.append(self.hds, new_hd)
            self.turns = np.append(self.turns, new_turn)

        return new_speed, new_hd
    
    def fetch_objects(self):
        objects = self.environment.objects
        return objects

    def check_for_objects(self, objects):
        self.x, self.y = self.positions[-1]
        #objects_id_seen = []
        n_obj = len(objects) #number of objects
        object_track_in_pose = np.zeros((1,n_obj)) #will track objects with current pose
        for i, object in enumerate(objects): 
            obj_id = object.id
            obj_x = object.x

            obj_y = object.y
            obj_pos = object.position

            #TODO! Check if there is an obstacle/wall between agent and object
            #      This should block the visual field and object should not be stored.
            alpha = np.arctan((obj_y-self.y)/(obj_x - self.x)) #angle between rat and object 
            dist = np.linalg.norm(self.positions[-1] - obj_pos)
            vfr_ang, vfr_r = self.vfr # visual field range

            if (dist <= vfr_r) and (abs(self.hds[-1] - alpha) <= vfr_ang):
                #objects_id_seen.append(obj_id)
                object_track_in_pose[0,i] = 1

        return object_track_in_pose #objects_id_seen

    def track_objects_seen(self):
        p_i = np.sum(self.objects_observed, axis = 1)>0 #Truth values where objects are detected
        pos_obj_seen = self.positions[p_i] #positions of agent where objects are seen
        obj_seen = self.objects_observed[p_i] #corresponding objects seen vectors 
        return pos_obj_seen, obj_seen


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
