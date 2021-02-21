import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.style.use('dark_background')
G = 1


class Interaction:
    def __init__(self, masses):
        # Initialise the system with multiple rings of varying test particle density.
        # Initially, particles will be evenly spaced along the ring

        self.masses = masses
        self.time_step = 0.1
        self.steps = 100000

        self.particle_density_factor = 1
        #[[2, 15], [3, 18], [4, 24], [5, 30], [6, 36]]
        self.radii_density = np.array([[10,  1]])

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
                solution = self.solving_odes(radius, 0, i * 2 * np.pi / particle_at_radius, 0.1)
                self.particle_solutions[particle_no] = solution
                particle_no += 1

    def accel(self, r, theta):
        pos_x, pos_y = r*np.cos(theta), r*np.sin(theta)
        accel_r, accel_theta = 0, 0
        for mass in self.masses:
            # Go through list of heavy masses and work out their contributions to the force on particles
            x, y = mass[1], mass[2]
            accel_mag = G * mass[0] / ((pos_x-x)**2 + (pos_y-y)**2)

            l = r - x * np.cos(theta) - y * np.sin(theta)  # Geometry fun
            d = x * np.sin(theta) - y * np.cos(theta)

            cos_phi = l/np.sqrt(d**2+l**2)
            sin_phi = d/np.sqrt(d**2+l**2)

            accel_r -= accel_mag * cos_phi
            accel_theta -= accel_mag * sin_phi

        return accel_r, accel_theta

    def pair_odes(self, pos_vel, t):
        y0 = pos_vel[0]
        y1 = pos_vel[1]
        y2 = pos_vel[2]
        y3 = pos_vel[3]
        # Divide by the radius in the angular force term so that we get the acceleration of theta
        return [y1, y0*y3**2 + self.accel(y0, y2)[0], y3, -2*y1*y3/y0 + (1/y0)*self.accel(y0, y2)[1]]

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
        fig, ax = plt.subplots(figsize=(6, 4))
        self.produce_orbits()
        for particle in self.particle_solutions:
            r, theta = particle[1], particle[2]
            ax.plot(r*np.cos(theta), r*np.sin(theta), color='red', lw=0.2)

        # Styling
        for mass in self.masses:
            ax.scatter(mass[1], mass[2])
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Trajectories based on a central force at the origin for circular orbits.')

        plt.show()


if __name__ == "__main__":
    newInt = Interaction([(1, 0, 0), (3, 4, 0)])
    newInt.plot_orbits()


