import MPSFAST
import numpy as np
import time

G = 6.67e-11
initial_heavy_1 = np.array([[0, 0, 0], [0, 0, 0]])
m1 = 5.683e26  # Initial heavy mass
initial_heavy_2 = np.array([[185000000, 0, 0], [0, np.sqrt(G*m1/(185000000)), 0]])
m2 = 3.75e19  # Perturbing galaxy

num_test = 10

# Initialise the test particle positions and velocities array
particle_pos = np.zeros((num_test, 3))
particle_vels = np.zeros((num_test, 3))

for i in range(num_test):
    # Initialisation of test particle locations and velocities
    radius = (i + 1) * (80000000 - 7000000) / num_test + 7000000
    theta = np.pi
    speed = np.sqrt(G * m1 / radius)
    particle_pos[i] = [radius * np.cos(theta), radius * np.sin(theta), 0]
    particle_vels[i] = [-speed * np.sin(theta), speed * np.cos(theta), 0]

start_time = time.time()
system = MPSFAST.MPSFAST(G, m1, m2, num_test, initial_heavy_1, initial_heavy_2, particle_pos, particle_vels)
system.produce_animation()
print("--- %s seconds ---" % (time.time() - start_time))