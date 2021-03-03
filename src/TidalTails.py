import MPSFAST
import numpy as np
import time

G = 1
initial_heavy_1 = np.array([[0, 0, 0], [0, 0, 0]])
m1 = 1  # Initial heavy mass
initial_heavy_2 = np.array([[20, 20, 0], [0, -np.sqrt(2*G*m1/(20*1.41)), 0]])
m2 = 1  # Perturbing galaxy

num_test = 50

# Initialise the test particle positions and velocities array
particle_pos = np.zeros((num_test, 3))
particle_vels = np.zeros((num_test, 3))

for i in range(num_test):
    # Initialisation of test particle locations and velocities
    radius = (i + 1) * (6 - 2) / num_test + 2
    theta = 0
    speed = np.sqrt(G * m1 / radius)
    particle_pos[i] = [radius * np.cos(theta), radius * np.sin(theta), 0]
    particle_vels[i] = [-speed * np.sin(theta), speed * np.cos(theta), 0]

start_time = time.time()
system = MPSFAST.MPSFAST(G, m1, m2, num_test, initial_heavy_1, initial_heavy_2, particle_pos, particle_vels)
system.produce_animation()
print("--- %s seconds ---" % (time.time() - start_time))