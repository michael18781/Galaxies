import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time
G = 1


class Interaction:
    def __init__(self, mass1):
        # Initialise the system with multiple rings of varying test particle density.
        # Initially, particles will be evenly spaced along the ring

        self.M = mass1
        self.time_step = 0.1
        self.steps = 10000

        self.particle_density_factor = 1
        self.radii_density = np.array([[2, 15], [3, 18], [4, 24], [5, 30], [6, 36]])

        # Total number of particles

        self.total_particles = int(self.particle_density_factor * np.sum(self.radii_density, axis=0)[1])
        self.particle_solutions = np.zeros((self.total_particles, 5, self.steps))  # Array for particle positions

    def produce_orbits(self):
        # Initialisation for circular orbits evenly spaced
        particle_no = 0
        for radii in self.radii_density:
            radius, particle_at_radius = radii[0], radii[1] * self.particle_density_factor
            for i in range(particle_at_radius):
                # Finding the solution
                solution = self.solving_odes(radius, 0, i * 2 * np.pi / particle_at_radius, np.sqrt(G * self.M / radius) / radius)
                self.particle_solutions[particle_no] = solution
                particle_no += 1

    def force(self, r, theta):
        return G * self.M/(r**2)

    def pair_odes(self, pos_vel, t):
        y0 = pos_vel[0]
        y1 = pos_vel[1]
        y2 = pos_vel[2]
        y3 = pos_vel[3]
        return [y1, y0*y3**2 - self.force(y0, y2), y3, -2*y1*y3/y0]

    def solving_odes(self, r0, r_dot0, theta0, theta_dot0):
        times = np.arange(0, self.steps * self.time_step, self.time_step)
        # Solution takes the pair of odes, initial conditions, and an array of times
        solution = integrate.odeint(self.pair_odes, [r0, r_dot0, theta0, theta_dot0], times)

        # Extracting individual solutions
        r = solution[:, 0]
        r_dot = solution[:, 1]
        theta = solution[:, 2]
        theta_dot = solution[:, 3]
        return np.array([times, r, theta, r_dot, theta_dot])

    def plot_orbits(self):
        # Test with first particle
        fig, ax = plt.subplots()
        self.produce_orbits()
        for particle in self.particle_solutions:
            r, theta = particle[1], particle[2]
            plt.plot(r*np.cos(theta), r*np.sin(theta), label="", color='black')

        plt.show()


if __name__ == "__main__":
    newInt = Interaction(1)
    newInt.plot_orbits()


