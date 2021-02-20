import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
sns.set_theme(style="whitegrid")


class Interaction:
    def __init__(self):
        # Initialise the system with multiple rings of varying test particle density.
        # Initially, particles will be evenly spaced along the ring

        particle_density_factor = 1
        self.anim_interval = 10
        self.time_interval = self.anim_interval * 0.001 # Animation interval in ms
        self.radii_density = np.array([[2, 15], [3, 18], [4, 24], [5, 30], [6, 36]])

        # Total number of particles
        self.total_particles = int(particle_density_factor * np.sum(self.radii_density, axis=0)[1])
        self.positions = np.zeros((self.total_particles, 3))  # Array for particle positions
        self.velocities = np.zeros((self.total_particles, 3))  # Array for particle velocities

        # Initialisation for circular orbits evenly spaced
        particle_no = 0
        for radii in self.radii_density:
            radius, particle_at_radius = radii[0], radii[1] * particle_density_factor
            for i in range(particle_at_radius):
                self.positions[particle_no] = np.array([radius, i * 2 * np.pi/particle_at_radius, 0])
                self.velocities[particle_no] = np.array([0, np.sqrt(1 / radius), 0])
                particle_no += 1

    def update_positions(self):
        self.positions += self.velocities * self.time_interval

    def plot(self, i):
        # Function simply plots the position of every particle
        plt.cla()
        plt.title(f'Plot of test particle motions, T = {self.time_interval * i:.2f}s')

        self.update_positions()
        r = self.positions[:, 0]  # Current particle radii
        theta = self.positions[:, 1]  # Current particle thetas
        sns.scatterplot(x=r * np.cos(theta), y=r * np.sin(theta), color='g')  # Plot cartesian position

    def animate(self):
        fig = plt.figure(figsize=(10, 10))
        ani = FuncAnimation(fig, self.plot, interval=self.anim_interval)
        plt.show()


if __name__ == "__main__":
    newInt = Interaction()
    newInt.animate()
