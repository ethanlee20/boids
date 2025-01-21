
"""Boids simulation"""


from random import uniform, sample
from math import sqrt, atan2, cos, sin, pi
from time import sleep


class Rect:

    """A rectangle."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def quantize(self):
        return Rect(int(self.width), int(self.height))


class Vec_2D:

    """
    A 2D vector. Theta in radians.

    Initialize with either x and y or r and theta.
    """

    def __init__(self, x=None, y=None, r=None, theta=None):
        if (x is not None) and (y is not None) and (r is None) and (theta is None):
            self.x = x
            self.y = y
            self.r = sqrt(x**2 + y**2)
            self.theta = atan2(y, x)
        elif (x is None) and (y is None) and (r is not None) and (theta is not None):
            self.x = r * cos(theta)
            self.y = r * sin(theta)
            self.r = r
            self.theta = theta

    def __add__(self, other):
        result_x = self.x + other.x
        result_y = self.y + other.y 
        return Vec_2D(x=result_x, y=result_y)

    def __sub__(self, other):
        result_x = self.x - other.x
        result_y = self.y - other.y
        return Vec_2D(x=result_x, y=result_y)

    def __rmul__(self, other):
        # Only real scalar multiplication implemented
        assert isinstance(other, float | int)
        result_x  = other * self.x
        result_y = other * self.y
        return Vec_2D(x=result_x, y=result_y)

    def normalized(self):
        return Vec_2D(r=1, theta=self.theta)

    def cart_quantize(self):
        return Vec_2D(int(self.x), int(self.y))

    def cart_tuple(self):
        return (self.x, self.y)


class Screen:

    """Text graphics visualizer."""

    def __init__(self, shape:Rect, empty_char=' '):
        self.empty_char = empty_char
        self.shape = shape.quantize()
        self.grid = self.create_empty_grid()

    def create_empty_grid(self):
        empty_grid = [
            [self.empty_char for col in range(self.shape.width)]
            for row in range(self.shape.height)
        ]
        return empty_grid

    def draw(self):
        output = '\n'.join([''.join(row) for row in self.grid])
        print(output)

    def clear_all(self):
        self.grid = self.create_empty_grid()

    def clear(self, pos:Vec_2D):
        pos = pos.cart_quantize()
        self.grid[pos.y][pos.x] = self.empty_char

    def get(self, pos:Vec_2D):
        pos = pos.cart_quantize()
        return self.grid[pos.y][pos.x]

    def set(self, pos:Vec_2D, data):
        pos = pos.cart_quantize()
        self.grid[pos.y][pos.x] = data

    def set_row(self, row, data):
        [self.set(Vec_2D(x=col, y=row), data) for col in range(self.shape.width)]

    def set_col(self, col, data):
        [self.set(Vec_2D(x=col, y=row), data) for row in range(self.shape.height)]

    def set_border(self):
        self.set_col(0, '|')
        self.set_col(self.shape.width-1, '|')
        self.set_row(0, '-')
        self.set_row(self.shape.height-1, '-')


class Boid:

    """
    A bird-oid.

    Parameters
    ----------
    pos_bounds : tuple
        (min, max) positions before applying boundary force,
        applies to x and y components.
    max_speed : float
        Max speed, before applying air resistance force.
    sight_radius : float
        Radius within which boid can see others.
    max_neighbors : int
        Maximum number of neighbors boid can see.
        Saves computation.
    separation_constant : float
        Constant (> 0) which determines strength of separation force.
    alignment_constant : float
        Constant (> 0) which determines strength of alignment force.
    cohesion_constant : float
        Constant (> 0) which determines strength of cohesion force.
    boundary_constant : float
        Constant (> 0) which determines strength of boundary force.
    air_resistance_constant : float
        Constant (> 0) which determines strength of air resistance force.
    """

    def __init__(
        self,
        pos_bounds:tuple,
        max_speed:float,
        sight_radius:float,
        max_neighbors:int,
        separation_constant:float,
        alignment_constant:float,
        cohesion_constant:float,
        boundary_constant:float,
        air_resistance_constant:float
    ):
        self.pos_bounds = pos_bounds
        self.max_speed = max_speed
        self.sight_radius = sight_radius
        self.max_neighbors = max_neighbors
        self.separation_constant = separation_constant
        self.alignment_constant = alignment_constant
        self.cohesion_constant = cohesion_constant
        self.boundary_constant = boundary_constant
        self.air_resistance_constant = air_resistance_constant
        self.pos = Vec_2D(
            x=uniform(*pos_bounds),
            y=uniform(*pos_bounds),
        )
        self.v = Vec_2D(
            r=uniform(0, max_speed),
            theta=uniform(-pi, pi),
        )

    def update(self, boids:list, timestep):
        self.update_v(boids, timestep)
        self.update_pos(timestep)

    def update_v(self, boids:list, timestep):
        neighbors = self.sense(boids)
        for b in neighbors:
            self.v += 1/len(neighbors) * timestep * self.compute_separation_force(b)
            self.v += 1/len(neighbors) * timestep * self.compute_alignment_force(b)
            self.v += 1/len(neighbors) * timestep * self.compute_cohesion_force(b)
        self.v += timestep * self.compute_boundary_force()
        self.v += timestep * self.compute_air_resistance_force()

    def update_pos(self, timestep):
        self.pos += timestep * self.v

    def sense(self, boids:list):
        neighbors = [
            b for b in boids
            if (b is not self)
            and ((b.pos - self.pos).r <= self.sight_radius)
        ]
        sample_size = (self.max_neighbors
            if (len(neighbors) >= self.max_neighbors)
            else len(neighbors)
        )
        return sample(neighbors, sample_size)

    def compute_separation_force(self, neighbor):
        disp = neighbor.pos - self.pos
        if disp.r < 1: # set effective minimum distance for stability
            disp = Vec_2D(r=1, theta=disp.theta)
        force = -self.separation_constant * 1/disp.r * disp.normalized()
        return force

    def compute_alignment_force(self, neighbor):
        delta_v = neighbor.v - self.v
        force = self.alignment_constant * delta_v
        return force

    def compute_cohesion_force(self, neighbor):
        disp = neighbor.pos - self.pos
        force = self.cohesion_constant * disp
        return force

    def compute_boundary_force(self):
        force_comp_x = force_comp_y = 0
        if self.pos.x < self.pos_bounds[0]:
            force_comp_x = self.boundary_constant * (self.pos_bounds[0] - self.pos.x)
        elif self.pos.x > self.pos_bounds[1]:
            force_comp_x = self.boundary_constant * (self.pos_bounds[1] - self.pos.x)
        if self.pos.y < self.pos_bounds[0]:
            force_comp_y = self.boundary_constant * (self.pos_bounds[0] - self.pos.y)
        elif self.pos.y > self.pos_bounds[1]:
            force_comp_y = self.boundary_constant * (self.pos_bounds[1] - self.pos.y)
        force = Vec_2D(x=force_comp_x, y=force_comp_y)
        return force

    def compute_air_resistance_force(self):
        force = Vec_2D(x=0, y=0)
        if self.v.r > self.max_speed:
            force = -self.air_resistance_constant * (self.v.r - self.max_speed)**2 * self.v.normalized()
        return force
        

class World:
    
    """The world!"""
    
    def __init__(self, n_boids:int, timestep:float, boid_params:dict):
        self.timestep = timestep
        self.boids = [
            Boid(**boid_params) for _ in range(n_boids)
        ]
        
    def update(self):
        for b in self.boids:
            b.update(self.boids, self.timestep)


def main():
    
    n_boids = 50 # number of boids
    timestep = 0.5 # in-simulation time between updates
    n_iter = 2000 # number of timesteps to run

    frame_time = 0.033 # about 30 fps
    boid_char = '*' # boid graphics

    boid_params = {
        "pos_bounds": (6, 25),
        "max_speed": 2,
        "sight_radius": 5,
        "max_neighbors": 2,
        "separation_constant": 2.3,
        "alignment_constant": 0.03,
        "cohesion_constant": 0.4,
        "boundary_constant": 0.1,
        "air_resistance_constant": 3,
    }

    screen_buffer_width = boid_params["pos_bounds"][0]
    screen_side_length = boid_params["pos_bounds"][1] + screen_buffer_width
    screen_shape = Rect(screen_side_length, screen_side_length)

    world = World(n_boids, timestep, boid_params)

    screen = Screen(screen_shape, empty_char=' ')

    for _ in range(n_iter):

        sleep(frame_time)

        world.update()

        screen.clear_all()
        screen.set_border()

        for b in world.boids:
            if (b.pos.x > 0) and (b.pos.y > 0):
                try: screen.set(b.pos, boid_char)
                except IndexError: pass
        screen.draw()

    
if __name__ == "__main__":
    main()
        
        




        

        
        
        
