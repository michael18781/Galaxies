import numpy as np
import scipy.integrate as integrate
import scipy as sp
import matplotlib.pyplot as plt
plt.style.use('dark_background')

softening_radius = 0.001  # Parameter to stop issues from particles being too close to each other


class MultipleParticleSystem:
    def __init__(self, mass1, mass2, number_test_particles, init_heavy_1, init_heavy_2):
        # The value of our initial parameters for two heavy particles.
        self.initial_r_1, self.initial_v_1 = init_heavy_1[0], init_heavy_1[1]
        self.initial_r_2, self.initial_v_2 = init_heavy_2[0], init_heavy_2[1]

        # Initial parameters.
        self.mass1, self.mass2 = mass1, mass2
        self.number_test_particles = number_test_particles

        # Initialise the test particle positions and velocities.
        self.particle_initials = np.array([[10, 0, 0], [0, 0.2, 0]])

    def equations_of_motion(self, pos_and_vels, t):
        # Equations of motion which will be used by the odeint solver. pos_and_vels is our total state vector which
        # describes every particle in the system.
        pos_heavy_1, vel_heavy_1 = pos_and_vels[:3], pos_and_vels[3:6]
        pos_heavy_2, vel_heavy_2 = pos_and_vels[6:9], pos_and_vels[9:12]

        # Creating an array which will house the differential equations governing the motion of the test particles.
        particle_derivatives = np.zeros((2 * self.number_test_particles, 3))

        r = sp.linalg.norm(pos_heavy_2 - pos_heavy_1) + softening_radius

        # Computing the rate of change of velocity and position for the heavy masses in vector form.
        vel_heavy_1_dot = self.mass2 * (pos_heavy_2 - pos_heavy_1) / r ** 3 if self.mass1 != 0.0 \
            else np.array([0.0, 0.0, 0.0])
        vel_heavy_2_dot = self.mass1 * (pos_heavy_1 - pos_heavy_2) / r ** 3 if self.mass2 != 0.0 \
            else np.array([0.0, 0.0, 0.0])

        pos_heavy_1_dot = vel_heavy_1
        pos_heavy_2_dot = vel_heavy_2

        # Differential equations defined for the heavy masses.
        heavy_derivatives = np.concatenate((pos_heavy_1_dot, vel_heavy_1_dot, pos_heavy_2_dot, vel_heavy_2_dot))

        # Defining differential equations for test masses.
        for i in range(self.number_test_particles):
            pos_particle = pos_and_vels[12 + 3 * i:15 + 3 * i]
            vel_particle = pos_and_vels[15 + 3 * i:18 + 3 * i]

            # Distance to heavy masses including softening radius.
            r_to_1 = sp.linalg.norm(pos_particle - pos_heavy_1) + softening_radius
            r_to_2 = sp.linalg.norm(pos_particle - pos_heavy_2) + softening_radius

            # Derivatives of velocity
            particle_derivatives[i + 1] = self.mass2 * (pos_heavy_2 - pos_particle) / (r_to_2 ** 3) + \
                                          self.mass1 * (pos_heavy_1 - pos_particle) / (r_to_1 ** 3)
            # Derivatives of position
            particle_derivatives[i] = vel_particle

        all_derivatives = heavy_derivatives
        for deriv in particle_derivatives:
            # Still have an array of shape (2 * num particles, 3) so need to put these all in together using for loop
            all_derivatives = np.concatenate((all_derivatives, deriv))

        return all_derivatives

    def produce_solution(self):
        initial_conditions = np.array([self.initial_r_1, self.initial_v_1, self.initial_r_2, self.initial_v_2])
        initial_conditions = np.concatenate((initial_conditions, self.particle_initials)).flatten()
        t = np.arange(0, 1000, 0.1)
        sol = integrate.odeint(self.equations_of_motion, initial_conditions, t)
        return sol

    def plot_solution(self):
        # Test with first particle
        fig, ax = plt.subplots(figsize=(6, 4))
        solution = self.produce_solution()

        pos_heavy1 = solution[:, :3]
        pos_heavy2 = solution[:, 6:9]

        for i in range(self.number_test_particles):
            position = solution[:, 12 + 3 * i:15 + 3 * i]
            ax.plot(position[:, 0], position[:, 1], color='red', lw=1)  # Full trajectory.
            ax.scatter(position[-1][0], position[-1][1], color='red', s=100)  # Last positions.

        # Full trajectory.
        ax.plot(pos_heavy1[:, 0], pos_heavy1[:, 1], color='blue', lw=1)
        ax.plot(pos_heavy2[:, 0], pos_heavy2[:, 1], color='black', lw=1)

        # Last positions.
        ax.scatter(pos_heavy2[-1][0], pos_heavy2[-1][1], color='black', s=100)
        ax.scatter(pos_heavy1[-1][0], pos_heavy1[-1][1], color='blue', s=100)

        # Styling
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Trajectories based on a central force at the origin for circular orbits.')
        plt.show()


if __name__ == "__main__":
    initial_heavy_1 = np.array([[0, 0, 0], [0, 0, 0]])
    initial_heavy_2 = np.array([[0, 0, 0], [0, 0, 0]])
    num_test = 1
    system = MultipleParticleSystem(0, 1, num_test, initial_heavy_1, initial_heavy_2)
    system.plot_solution()
