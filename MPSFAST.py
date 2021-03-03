import numpy as np
import scipy.integrate as integrate
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')


class MPSFAST:
    def __init__(self, G, mass1, mass2, number_test_particles, init_heavy_1, init_heavy_2, particle_pos, particle_vels):
        # The value of our initial parameters for two heavy particles.
        self.initial_r_1, self.initial_v_1 = init_heavy_1[0], init_heavy_1[1]
        self.initial_r_2, self.initial_v_2 = init_heavy_2[0], init_heavy_2[1]

        # Initial parameters.
        self.mass1, self.mass2 = mass1, mass2
        self.number_test_particles = number_test_particles

        self.position_inits = np.concatenate((self.initial_r_1, self.initial_r_2, particle_pos.flatten()))
        self.velocity_inits = np.concatenate((self.initial_v_1, self.initial_v_2, particle_vels.flatten()))
        self.initials = np.concatenate((self.position_inits, self.velocity_inits))

        self.heavy_radius = 0.1
        self.test_radius = 0.01
        self.G = G
        self.softening_radius = 0.01
        self.collision = False

    def equations_of_motion(self, pos_and_vels, t):
        # Equations of motion which will be used by the odeint solver. pos_and_vels is our total state vector which
        # describes every particle in the system.
        positions, velocities = np.array_split(pos_and_vels, 2)
        pos_heavy_1, vel_heavy_1 = positions[:3], velocities[:3]
        pos_heavy_2, vel_heavy_2 = positions[3:6], velocities[3:6]

        r = sp.linalg.norm(pos_heavy_2 - pos_heavy_1) + self.softening_radius

        if r < self.heavy_radius * 2:
            # Heavy masses have collided, combine into mass 1 and continue.
            self.collision = True
            self.mass1 += self.mass2
            self.mass2 = 0

        # Computing the rate of change of velocity and position for the heavy masses in vector form.
        vel_heavy_1_dot = self.G * self.mass2 * (pos_heavy_2 - pos_heavy_1) / (r ** 3) if self.mass1 != 0 else np.array([0, 0, 0])
        vel_heavy_2_dot = self.G * self.mass1 * (pos_heavy_1 - pos_heavy_2) / (r ** 3) if self.mass2 != 0 else np.array([0, 0, 0])

        pos_heavy_1_dot = vel_heavy_1
        pos_heavy_2_dot = vel_heavy_2

        particle_positions = np.split(positions[6:], positions[6:].size//3)
        particle_velocities = np.split(velocities[6:], velocities[6:].size//3)

        # Distance to heavy masses including softening radius.
        r_to_1 = np.linalg.norm(particle_positions - pos_heavy_1, axis=1, keepdims=True) + self.softening_radius
        r_to_2 = np.linalg.norm(particle_positions - pos_heavy_2, axis=1, keepdims=True) + self.softening_radius

        # Derivatives of velocity
        particle_vel_dot = np.divide(self.G * self.mass2 * (pos_heavy_2 - particle_positions), r_to_2 ** 3) \
                           + np.divide(self.G * self.mass1 * (pos_heavy_1 - particle_positions), r_to_1 ** 3)
        # Derivatives of position
        particle_pos_dot = particle_velocities

        position_derivs = np.zeros((self.number_test_particles+2, 3))
        velocity_derivs = np.zeros((self.number_test_particles + 2, 3))

        position_derivs[0], position_derivs[1] = pos_heavy_1_dot, pos_heavy_2_dot
        velocity_derivs[0], velocity_derivs[1] = vel_heavy_1_dot, vel_heavy_2_dot

        position_derivs[2:] = particle_pos_dot
        velocity_derivs[2:] = particle_vel_dot

        return np.concatenate((position_derivs.flatten(), velocity_derivs.flatten()))

    def produce_solution(self):
        t = np.arange(0, 100000, 0.5)
        sol = integrate.odeint(self.equations_of_motion, self.initials, t)
        return sol

    def static_plot(self):
        # Test with first particle
        fig, ax = plt.subplots(figsize=(6, 4))
        solution = self.produce_solution()
        positions = np.array_split(solution, 2)[0]

        pos_heavy1 = positions[:, :3]
        pos_heavy2 = positions[:, 3:6]

        for i in range(self.number_test_particles):
            position = positions[:, 3*i+6:3*i+9]
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
            # i.e. even line index so it must be a trajectory plot
            line.set_data(data[j][:i + 1][:, 0], data[j][:i + 1][:, 1])

        for (j, scat) in enumerate(scatters):
            scat.set_data(data[j][:i + 1][:, 0][-1], data[j][:i + 1][:, 1][-1])

        return lines+scatters

    def produce_animation(self):
        sol = self.produce_solution()
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(xlim=(-2e8, 2e8), ylim=(-2e8, 2e8))

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Trajectories based on a central force at the origin for circular orbits.')

        lines = []  # List to hold line elements
        scatters = []  # List to hold scatter plots
        data = []  # List to hold data arrays

        data1 = sol[:, :3]
        data.append(data1)
        data2 = sol[:, 3:6]
        data.append(data2)

        for i in range(self.number_test_particles):
            particle_data = sol[:, 3*i+6:3*i+9]
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

        fps = 60

        ani = FuncAnimation(fig, self.animate, fargs=(lines, scatters, data), frames=10000, interval=1000/fps, blit=True)
        plt.show()
