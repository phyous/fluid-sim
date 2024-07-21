import numpy as np
import pygame
from pygame.math import Vector3
from OpenGL import GL
from OpenGL.GL import *
from OpenGL.GLU import *

# SPH parameters
PARTICLE_MASS = 0.02
H = 0.1  # Smoothing length
GAS_CONSTANT = 2000.0
REST_DENSITY = 1000.0
VISCOSITY = 250.0
GRAVITY = Vector3(0, -9.8, 0)
DT = 0.0008
EPSILON = 1e-6  # Small value to prevent division by zero

class Particle:
    def __init__(self, pos):
        self.pos = Vector3(pos)
        self.vel = Vector3(0, 0, 0)
        self.force = Vector3(0, 0, 0)
        self.density = 0.0
        self.pressure = 0.0

class Grid:
    def __init__(self, width, height, depth, cell_size):
        self.cell_size = cell_size
        self.width = int(width / cell_size) + 1
        self.height = int(height / cell_size) + 1
        self.depth = int(depth / cell_size) + 1
        self.cells = [[] for _ in range(self.width * self.height * self.depth)]

    def get_cell_index(self, pos):
        x = int(pos.x / self.cell_size)
        y = int(pos.y / self.cell_size)
        z = int(pos.z / self.cell_size)
        return x + y * self.width + z * self.width * self.height

    def add_particle(self, particle):
        cell_index = self.get_cell_index(particle.pos)
        self.cells[cell_index].append(particle)

    def clear(self):
        for cell in self.cells:
            cell.clear()

class Terrain:
    def __init__(self, width, depth, resolution):
        self.width = width
        self.depth = depth
        self.resolution = resolution
        self.heightmap = self.generate_crater_heightmap()

    def generate_crater_heightmap(self):
        x = np.linspace(-self.width/2, self.width/2, self.resolution)
        z = np.linspace(-self.depth/2, self.depth/2, self.resolution)
        X, Z = np.meshgrid(x, z)
        
        D = np.sqrt(X**2 + Z**2)
        Y = 5 * (1 - np.exp(-0.1 * D**2)) + 0.5 * np.sin(D)
        
        return Y

    def get_height(self, x, z):
        x_index = int((x + self.width/2) / self.width * (self.resolution - 1))
        z_index = int((z + self.depth/2) / self.depth * (self.resolution - 1))
        x_index = max(0, min(x_index, self.resolution - 1))
        z_index = max(0, min(z_index, self.resolution - 1))
        return self.heightmap[z_index, x_index]

    def render(self):
        glBegin(GL_TRIANGLES)
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                x1, z1 = i * self.width / (self.resolution - 1) - self.width/2, j * self.depth / (self.resolution - 1) - self.depth/2
                x2, z2 = (i+1) * self.width / (self.resolution - 1) - self.width/2, j * self.depth / (self.resolution - 1) - self.depth/2
                x3, z3 = i * self.width / (self.resolution - 1) - self.width/2, (j+1) * self.depth / (self.resolution - 1) - self.depth/2
                x4, z4 = (i+1) * self.width / (self.resolution - 1) - self.width/2, (j+1) * self.depth / (self.resolution - 1) - self.depth/2

                y1, y2, y3, y4 = self.heightmap[j,i], self.heightmap[j,i+1], self.heightmap[j+1,i], self.heightmap[j+1,i+1]

                glColor3f(0.6, 0.4, 0.2)
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
                glVertex3f(x3, y3, z3)

                glVertex3f(x2, y2, z2)
                glVertex3f(x4, y4, z4)
                glVertex3f(x3, y3, z3)
        glEnd()

class SPHSimulation:
    def __init__(self, width, height, depth, terrain):
        self.width = width
        self.height = height
        self.depth = depth
        self.terrain = terrain
        self.particles = []
        self.grid = Grid(width, height, depth, H)

    def initialize_particles(self, num_particles):
        for _ in range(num_particles):
            while True:
                pos = (np.random.rand(3) * [self.width, self.height/4, self.depth] - [self.width/2, 0, self.depth/2]).tolist()
                terrain_height = self.terrain.get_height(pos[0], pos[2])
                if terrain_height < pos[1] < terrain_height + self.height/8:
                    self.particles.append(Particle(pos))
                    break

    def update_grid(self):
        self.grid.clear()
        for particle in self.particles:
            self.grid.add_particle(particle)

    def get_neighbors(self, particle):
        cell_index = self.grid.get_cell_index(particle.pos)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell_index = cell_index + dx + dy * self.grid.width + dz * self.grid.width * self.grid.height
                    if 0 <= neighbor_cell_index < len(self.grid.cells):
                        neighbors.extend(self.grid.cells[neighbor_cell_index])
        return neighbors

    def compute_density_pressure(self):
        for p in self.particles:
            p.density = PARTICLE_MASS * self.cubic_kernel(0, H)  # Self-density
            for neighbor in self.get_neighbors(p):
                if p != neighbor:
                    r = p.pos.distance_to(neighbor.pos)
                    if r < H:
                        p.density += PARTICLE_MASS * self.cubic_kernel(r, H)
            p.pressure = GAS_CONSTANT * (p.density - REST_DENSITY)

    def compute_forces(self):
        for p in self.particles:
            p.force = Vector3(0, 0, 0)
            for neighbor in self.get_neighbors(p):
                if p != neighbor:
                    r = p.pos - neighbor.pos
                    dist = r.length()
                    if dist < H and dist > 0:  # Added check for dist > 0
                        pressure_force = -PARTICLE_MASS * (p.pressure + neighbor.pressure) / (2 * max(neighbor.density, EPSILON)) * self.gradient_kernel(r, H)
                        viscosity_force = VISCOSITY * PARTICLE_MASS * (neighbor.vel - p.vel) / max(neighbor.density, EPSILON) * self.laplacian_kernel(dist, H)
                        p.force += pressure_force + viscosity_force
            p.force += GRAVITY * PARTICLE_MASS

    def integrate(self):
        for p in self.particles:
            if p.density > EPSILON:
                p.vel += p.force / p.density * DT
            new_pos = p.pos + p.vel * DT
            
            terrain_height = self.terrain.get_height(new_pos.x, new_pos.z)
            if new_pos.y < terrain_height:
                normal = Vector3(0, 1, 0)
                p.vel = p.vel - 2 * p.vel.dot(normal) * normal
                new_pos.y = terrain_height
            
            p.pos = new_pos

            p.pos.x = max(-self.width/2, min(p.pos.x, self.width/2))
            p.pos.y = max(0, min(p.pos.y, self.height))
            p.pos.z = max(-self.depth/2, min(p.pos.z, self.depth/2))

    def update(self):
        self.update_grid()
        self.compute_density_pressure()
        self.compute_forces()
        self.integrate()

    def apply_earthquake(self):
        for p in self.particles:
            p.vel += Vector3(np.random.uniform(-5, 5), np.random.uniform(0, 10), np.random.uniform(-5, 5))

    def render(self):
        positions = np.array([p.pos for p in self.particles], dtype=np.float32)
        colors = np.zeros((len(self.particles), 4), dtype=np.float32)
        
        for i, particle in enumerate(self.particles):
            depth = (particle.pos.y - self.terrain.get_height(particle.pos.x, particle.pos.z)) / 5
            refraction = min(1.0, 0.2 + depth * 0.1)
            colors[i] = [0.0, 0.2, 0.5, refraction]  # Deeper blue color

        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(3)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)

        GL.glVertexPointer(3, GL.GL_FLOAT, 0, positions)
        GL.glColorPointer(4, GL.GL_FLOAT, 0, colors)

        GL.glDrawArrays(GL.GL_POINTS, 0, len(self.particles))

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)

        GL.glDisable(GL.GL_POINT_SMOOTH)

    @staticmethod
    def cubic_kernel(r, h):
        q = r / h
        if q <= 1:
            return 15 / (2 * np.pi * h**3) * (2/3 - q**2 + 0.5 * q**3)
        elif q <= 2:
            return 15 / (2 * np.pi * h**3) * (1/6) * (2 - q)**3
        return 0

    @staticmethod
    def gradient_kernel(r, h):
        r_len = r.length()
        if r_len > EPSILON:
            q = r_len / h
            if q <= 1:
                return r.normalize() * 15 / (np.pi * h**4) * (-2 * q + 1.5 * q**2)
            elif q <= 2:
                return r.normalize() * 15 / (np.pi * h**4) * (-0.5 * (2 - q)**2)
        return Vector3(0, 0, 0)

    @staticmethod
    def laplacian_kernel(r, h):
        q = r / h
        if q <= 1:
            return 45 / (np.pi * h**5) * (1 - q)
        elif q <= 2:
            return 45 / (np.pi * h**5) * (2 - q) * 1/6
        return 0

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, -5.0, -20)

    terrain = Terrain(20, 20, 50)
    simulation = SPHSimulation(20, 20, 20, terrain)
    simulation.initialize_particles(20000)  # Increased number of particles

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    clock = pygame.time.Clock()
    running = True
    
    # Variables for rotation
    rotation_x, rotation_y = 0, 0
    last_mouse = pygame.mouse.get_pos()
    mouse_pressed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pressed = True
                    last_mouse = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    mouse_pressed = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_pressed:
                    new_mouse = pygame.mouse.get_pos()
                    dx = new_mouse[0] - last_mouse[0]
                    dy = new_mouse[1] - last_mouse[1]
                    rotation_y += dx * 0.5
                    rotation_x += dy * 0.5
                    last_mouse = new_mouse
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation.apply_earthquake()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        terrain.render()
        simulation.update()
        simulation.render()

        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()