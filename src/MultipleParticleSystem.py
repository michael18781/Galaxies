import numpy as np
import scipy.integrate as integrate
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

plt.style.use('dark_background')
G = 1


class MultipleParticleSystem:
    def __init__(self, mass1, mass2, number_test_particles, init_heavy_1, init_heavy_2, initial_particles):
        # The value of our initial parameters for two heavy particles.
        self.initial_r_1, self.initial_v_1 = init_heavy_1[0], init_heavy_1[1]
        self.initial_r_2, self.initial_v_2 = init_heavy_2[0], init_heavy_2[1]

        # Initial parameters.
        self.mass1, self.mass2 = mass1, mass2
        self.number_test_particles = number_test_particles

        self.particle_initials = initial_particles

        self.heavy_radius = 0.1
        self.test_radius = 0.01
        self.softening_radius = 0.01
        self.collision = False

    def equations_of_motion(self, pos_and_vels, t):
        # Equations of motion which will be used by the odeint solver. pos_and_vels is our total state vector which
        # describes every particle in the system.
        pos_heavy_1, vel_heavy_1 = pos_and_vels[:3], pos_and_vels[3:6]
        pos_heavy_2, vel_heavy_2 = pos_and_vels[6:9], pos_and_vels[9:12]

        r = sp.linalg.norm(pos_heavy_2 - pos_heavy_1) + self.softening_radius

        if r < self.heavy_radius * 2:
            # Heavy masses have collided, combine into mass 1 and continue.
            self.collision = True
            self.mass2 = 0

        # Computing the rate of change of velocity and position for the heavy masses in vector form.
        vel_heavy_1_dot = G * self.mass2 * (pos_heavy_2 - pos_heavy_1) / (r ** 3)
        vel_heavy_2_dot = G * self.mass1 * (pos_heavy_1 - pos_heavy_2) / (r ** 3)
        pos_heavy_1_dot = vel_heavy_1
        pos_heavy_2_dot = vel_heavy_2

        # Differential equations defined for the heavy masses.
        heavy_derivatives = np.concatenate((pos_heavy_1_dot, vel_heavy_1_dot, pos_heavy_2_dot, vel_heavy_2_dot))

        # Creating an array which will house the differential equations governing the motion of the test particles.
        particle_derivatives = np.zeros((2 * self.number_test_particles, 3))

        # Defining differential equations for test masses.
        for j in range(self.number_test_particles):
            pos_particle = pos_and_vels[12 + 6 * j:15 + 6 * j]
            vel_particle = pos_and_vels[15 + 6 * j:18 + 6 * j]

            # Distance to heavy masses including softening radius.
            r_to_1 = sp.linalg.norm(pos_particle - pos_heavy_1) + self.softening_radius
            r_to_2 = sp.linalg.norm(pos_particle - pos_heavy_2) + self.softening_radius

            if r_to_1 < self.heavy_radius + self.test_radius or r_to_2 < self.heavy_radius + self.test_radius:
                particle_derivatives[2*j + 1] = np.array([0, 0, 0])
                particle_derivatives[2 * j] = vel_particle
            else:
                # Derivatives of velocity
                particle_derivatives[2*j + 1] = G * self.mass2 * (pos_heavy_2 - pos_particle) / (r_to_2 ** 3) + \
                                              G * self.mass1 * (pos_heavy_1 - pos_particle) / (r_to_1 ** 3)
                # Derivatives of position
                particle_derivatives[2 * j] = vel_particle

        all_derivatives = heavy_derivatives

        for deriv in particle_derivatives:
            # Still have an array of shape (2 * num particles, 3) so need to put these all in together using for loop
            all_derivatives = np.concatenate((all_derivatives, deriv))

        return all_derivatives

    def produce_solution(self):
        initial_conditions = np.array([self.initial_r_1, self.initial_v_1, self.initial_r_2, self.initial_v_2])
        initial_conditions = np.concatenate((initial_conditions, self.particle_initials)).flatten()
        t = np.arange(0, 1000, 0.5)
        sol = integrate.odeint(self.equations_of_motion, initial_conditions, t)
        return sol

    def static_plot(self):
        # Test with first particle
        fig, ax = plt.subplots(figsize=(6, 4))
        solution = self.produce_solution()

        pos_heavy1 = solution[:, :3]
        pos_heavy2 = solution[:, 6:9]

        for i in range(self.number_test_particles):
            position = solution[:, 12 + 6 * i:15 + 6 * i]
            ax.plot(position[:, 0], position[:, 1], color='red', lw=1)  # Full trajectory.
            ax.scatter(position[-1][0], position[-1][1], color='red', s=100)  # Last positions.

        # Full trajectory.
        ax.plot(pos_heavy1[:, 0], pos_heavy1[:, 1], color='blue', lw=0.2)

        # Last position.
        ax.scatter(pos_heavy1[-1][0], pos_heavy1[-1][1], color='blue', s=100)

        if self.collision:
            ax.plot(pos_heavy2[:, 0], pos_heavy2[:, 1], color='black', lw=0.2)  # Trajectory.
            ax.scatter(pos_heavy2[-1][0], pos_heavy2[-1][1], color='black', s=100)  # Last position.
        else:
            ax.plot(pos_heavy2[:, 0], pos_heavy2[:, 1], color='white', lw=0.2)
            ax.scatter(pos_heavy2[-1][0], pos_heavy2[-1][1], color='white', s=100)

        # Styling
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Trajectories based on a central force at the origin for circular orbits.')
        plt.show()

    def animate(self, i, lines, scatters, data):
        for (j, line) in enumerate(lines):
            # i.e. Setting our trajectory plot
            line.set_data(data[j][:i + 1][:, 0], data[j][:i + 1][:, 1])

        for (j, scat) in enumerate(scatters):
            # Setting scatter plot for the end points of particles
            scat.set_data(data[j][:i + 1][:, 0][-1], data[j][:i + 1][:, 1][-1])

        return lines+scatters

    def produce_animation(self):
        sol = self.produce_solution()
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50))

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Trajectories based on a central force at the origin for circular orbits.')

        lines = []  # List to hold line elements
        scatters = []  # List to hold scatter plots
        data = []  # List to hold data arrays

        data1 = sol[:, :3]
        data.append(data1)
        data2 = sol[:, 6:9]
        data.append(data2)

        for i in range(self.number_test_particles):
            particle_data = sol[:, 12 + 6 * i:15 + 6 * i]
            data.append(particle_data)

        heavy1 = ax.plot([], [], color='blue', lw=0.5)[0]
        heavy1blob = ax.plot([], [], "o", color='blue')[0]
        lines.append(heavy1)
        scatters.append(heavy1blob)

        if self.collision:
            heavy2 = ax.plot([], [], color='black', lw=0.5)[0]
            heavy2blob = ax.plot([], [], "o", color='black')[0]
        else:
            heavy2 = ax.plot([], [], color='white', lw=0.5)[0]
            heavy2blob = ax.plot([], [], "o", color='white')[0]
        lines.append(heavy2)
        scatters.append(heavy2blob)

        for i in range(self.number_test_particles):
            particle_line = ax.plot([], [], color='red', lw=0.5)[0]
            particle_blob = ax.plot([], [], marker="o", color='red', markersize=2)[0]
            lines.append(particle_line)
            scatters.append(particle_blob)

        # Produce animation at 10fps
        ani = FuncAnimation(fig, self.animate, fargs=(lines, scatters, data), frames=10000, interval=1000/10, blit=True)
        plt.show()

