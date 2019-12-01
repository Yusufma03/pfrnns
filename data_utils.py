import random
import math
import numpy as np


class Maze(object):
    def __init__(self, maze):
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)
        self.blocks = []
        self.update_cnt = 0
        self.beacons = []
        for y, line in enumerate(self.maze):
            for x, block in enumerate(line):
                if block:
                    nb_y = self.height - y - 1
                    self.blocks.append((x, nb_y))
                    if block == 2:
                        self.beacons.extend(((x, nb_y), (x + 1, nb_y), (x, nb_y + 1), (x + 1, nb_y + 1)))

    def is_in(self, x, y):
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        return True

    def is_free(self, x, y):
        if not self.is_in(x, y):
            return False

        yy = self.height - int(y) - 1
        xx = int(x)
        return self.maze[yy][xx] == 0

    def random_place(self):
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return x, y

    def random_free_place(self):
        while True:
            x, y = self.random_place()
            if self.is_free(x, y):
                return x, y

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance_to_beacons(self, x, y, obs_num=5):
        distances = []
        for c_x, c_y in self.beacons:
            d = self.distance(c_x, c_y, x, y)
            distances.append(d)
        return sorted(distances)[:obs_num]


def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]


def add_noise_gauss(level, *coords):
    return [x + np.random.normal(scale=level) for x in coords]
    # return [x + random.uniform(-level, level) for x in coords]


def add_little_noise(*coords):
    return add_noise(0.02, *coords)


def add_some_noise(*coords):
    return add_noise(0.1, *coords)


class Point(object):
    def __init__(self, x, y, heading=None, w=1, noisy=False):
        if heading is None:
            heading = random.uniform(0, 360)
        if noisy:
            x, y, heading = add_some_noise(x, y, heading)

        self.x = x
        self.y = y
        self.h = heading
        self.w = w

    def __repr__(self):
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w)

    @property
    def xy(self):
        return self.x, self.y

    @property
    def xyh(self):
        return self.x, self.y, self.h

    @classmethod
    def create_random(cls, count, maze):
        return [cls(*maze.random_free_place()) for _ in range(0, count)]

    def read_sensor(self, maze, obs_num):
        return maze.distance_to_beacons(*self.xy, obs_num)

    def advance_by(self, speed, checker=None, noisy=False):
        h = self.h
        if noisy:
            speed, h = add_little_noise(speed, h)
            h += random.uniform(-3, 3)  # needs more noise to disperse better
        r = math.radians(h)
        dx = math.sin(r) * speed
        dy = math.cos(r) * speed
        if checker is None or checker(self, dx, dy):
            self.move_by(dx, dy)
            return True
        return False

    def move_by(self, x, y):
        self.x += x
        self.y += y


class Robot(Point):
    def __init__(self, maze, speed=0.2):
        super(Robot, self).__init__(*maze.random_free_place(), heading=90)
        self.chose_random_direction()
        self.step_count = 0
        self.speed = speed

    def chose_random_direction(self):
        heading = random.uniform(0, 360)
        self.h = heading

    def read_sensor(self, maze, obs_num):
        obs = super(Robot, self).read_sensor(maze, obs_num)
        level = 0.1
        return [x + random.uniform(-level, level) for x in obs]

    def move(self, maze):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            if self.advance_by(self.speed, noisy=True,
                               checker=lambda r, dx, dy: maze.is_free(r.x + dx, r.y + dy)):
                break
            # Bumped into something or too long in same direction,
            # chose random new direction
            self.chose_random_direction()


def gen_traj(traj_len=100, obs_num=5):
    maze_data = np.loadtxt('maze.csv', delimiter=',')

    world = Maze(maze_data)

    speed = 0.2
    robbie = Robot(world, speed=speed)
    traj_ret = []

    for _ in range(traj_len):
        r_d = robbie.read_sensor(world, obs_num)
        step_data = [robbie.x, robbie.y, robbie.h]
        old_heading = robbie.h
        old_x = robbie.x
        old_y = robbie.y
        robbie.move(world)
        d_h = robbie.h - old_heading
        d_x = robbie.x - old_x
        d_y = robbie.y - old_y

        action = [d_x, d_y, d_h]

        step_data = step_data + action + r_d

        traj_ret.append(step_data)

    return np.array(traj_ret), maze_data


def gen_data(num_trajs, traj_len=50, obs_num=5):
    data = {
        'trajs': []
    }

    from tqdm import tqdm
    for _ in tqdm(range(num_trajs)):
        traj_data, maze = gen_traj(traj_len, obs_num)
        data['trajs'].append(traj_data)

    data['map'] = maze

    return data