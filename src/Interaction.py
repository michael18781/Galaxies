import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.style.use('dark_background')
G = 1


class Interaction:
    def __init__(self, mass, particles, steps, duration):
        # Initialising system parameters
        self.mass = mass
        self.steps = steps
        self.duration = duration
        self.softening_radius = 0.001

        # Test particle set-up. Calculating correct sizes for our solutions array.
        self.particle_setup = particles
        self.total_particles = int(np.sum(self.particle_setup, axis=0)[1])

        # Array for particle positions
        self.particle_solutions = np.zeros((self.total_particles, 5, self.steps))
        # Array for particle energies
        self.particle_energies = np.zeros((self.total_particles, 2, self.steps))

    def accelerations(self, x, y):
        # Work out the contributions to the force on particles
        mass_x, mass_y = self.mass[1], self.mass[2]
        distance_to_mass = np.sqrt((mass_x - x) ** 2 + (mass_y - y) ** 2 + self.softening_radius ** 2)
        acceleration = G * self.mass[0] * np.array([(mass_x - x) / (distance_to_mass ** 3), (mass_y - y) / (distance_to_mass ** 3)])
        return acceleration

    def pair_odes(self, pos_vel, t):
        x, y, x_dot, y_dot = pos_vel[0], pos_vel[1], pos_vel[2], pos_vel[3]
        return [x_dot, y_dot, self.accelerations(x, y)[0], self.accelerations(x, y)[1]]

    def solving_odes(self, x0, y0, x_dot0, y_dot0):
        times = np.arange(0, self.duration, self.duration / self.steps)
        # Solution takes the pair of odes, initial conditions, and an array of times
        solution = integrate.odeint(self.pair_odes, [x0, y0, x_dot0, y_dot0], times)

        # Extracting individual solutions
        x, y, x_dot, y_dot = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]
        return np.array([times, x, y, x_dot, y_dot])

    def produce_orbits(self):
        # Initialisation for circular orbits evenly spaced
        particle_no = 0
        for radii in self.particle_setup:
            radius, particle_at_radius = radii[0], radii[1]
            for i in range(particle_at_radius):
                # Finding the solution
                theta = i * 2 * np.pi / particle_at_radius
                # Setting up circular orbits
                solution = self.solving_odes(radius * np.cos(theta), radius*np.sin(theta), 0, np.sqrt(G * self.mass[0] / radius))
                self.particle_solutions[particle_no] = solution
                particle_no += 1

        # Calculating the energies of the particles in orbits.
        for (i, particle_solution) in enumerate(self.particle_solutions):
            time = particle_solution[0]
            x, y = particle_solution[1], particle_solution[2]
            x_dot, y_dot = particle_solution[3], particle_solution[4]
            mass_x, mass_y = self.mass[1], self.mass[2]

            # Calculating potential and kinetic energy.
            gpe = -self.mass[0]/np.sqrt((x - mass_x) ** 2+(y - mass_y) ** 2)
            ke = 0.5 * (x_dot ** 2 + y_dot ** 2)

            # Setting the energies in the particle energy array.
            self.particle_energies[i] = np.array([time, gpe+ke])

    def plot_orbits(self):
        # Setting up figure and axes
        fig, ax = plt.subplots(figsize=(9, 6))
        self.produce_orbits()

        # Plotting trajectories of each particle
        for (i, particle) in enumerate(self.particle_solutions):
            x, y = particle[1], particle[2]
            ax.plot(x, y, color='red', lw=0.2)

        # Adding heavy mass
        ax.scatter(self.mass[1], self.mass[2])

        # Styling
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Trajectories based on a central force at the origin for circular orbits.')
        plt.show()

    def plot_energy(self):
        # Plotting energy of first particle
        fig, ax = plt.subplots(figsize=(9, 6))
        self.produce_orbits()

        # Plotting energy of each particle
        for particle in self.particle_energies:
            time = particle[0]
            energy = particle[1]
            ax.plot(time, energy)

        ax.set_title('Energy of the particle as a function of time.')
        ax.set_ylabel('Energy per unit mass')
        ax.set_xlabel('Time')
        plt.show()


interact = Interaction(mass=[3, 0, 0],
                       particles=[[6, 1], [5, 1]],
                       steps=95000,
                       duration=9500
                       )
interact.plot_orbits()


