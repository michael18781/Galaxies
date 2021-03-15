import MpsFast as MPS
import numpy as np
import time

# Reference quantities (all in SI system)
G = 1
mass_unit = 1
r_unit = 1
v_unit = np.sqrt(G * mass_unit / r_unit)
period_unit = 100

initial_heavy_1 = np.array([[0, 0, 0], [0, 0, 0]])
m1 = 1  # First galaxy.
initial_heavy_2 = np.array([[5, -5, 0], [0, np.sqrt(1/(5*1.41)), 0]])
m2 = 0  # Interacting galaxy
num_test = 20

# Initialise the test particle positions and velocities array
particle_pos = np.zeros((num_test, 3))
particle_vels = np.zeros((num_test, 3))

for i in range(num_test):
    # Initialisation of test particle locations and velocities
    radius = ((i + 1) * (6 - 1) / num_test + 1)/r_unit
    theta = np.pi
    speed = np.sqrt(1/radius)
    particle_pos[i] = [radius, 0, 0]
    particle_vels[i] = [0, speed, 0]

start_time = time.time()
system = MPS.MpsFast([G, mass_unit, r_unit, v_unit, period_unit], [m1, m2], num_test,
                     initial_heavy_1, initial_heavy_2, particle_pos, particle_vels)
system.static_plot()
print("--- %s seconds ---" % (time.time() - start_time))