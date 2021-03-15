import MpsFast as MPS
import numpy as np
import time
import random
from win10toast import ToastNotifier

# Reference quantities (all in SI system)
G = 6.67e-11
mass_mimas = 5.683e26
r_mimas = 185e6
v_mimas = np.sqrt(G*mass_mimas/r_mimas)
period_mimas = 0.942*86400

initial_heavy_1 = np.array([[0, 0, 0], [0, 0, 0]])
m1 = 1  # Saturn
initial_heavy_2 = np.array([[1, 0, 0], [0, 1, 0]])
m2 = 6.60e-4  # Mimas
num_test = 300

# Initialise the test particle positions and velocities array
particle_pos = np.zeros((num_test, 3))
particle_vels = np.zeros((num_test, 3))

for i in range(num_test):
    # Initialisation of test particle locations and velocities
    radius = ((i + 1) * (80e6 - 7e6) / num_test + 7e6)/r_mimas
    theta = random.random() * 2 * np.pi
    speed = np.sqrt(1 / radius)
    particle_pos[i] = [radius*np.cos(theta), radius*np.sin(theta), 0]
    particle_vels[i] = [-speed*np.sin(theta), speed*np.cos(theta), 0]

start_time = time.time()
system = MPS.MpsFast(scaling_constants=[G, mass_mimas, r_mimas, v_mimas, period_mimas],
                     heavy_masses=[m1, m2],
                     init_heavy_1=initial_heavy_1,
                     init_heavy_2=initial_heavy_2,
                     particle_pos=particle_pos,
                     particle_vels=particle_vels,
                     number_test_particles=num_test,
                     duration=10,
                     steps=10000
                     )

system.static_plot()
duration = time.time() - start_time
print("--- %s seconds ---" % duration)
toaster = ToastNotifier()
toaster.show_toast("Code has run.", f"Took {duration:.2f} seconds")
