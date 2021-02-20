import numpy as np
import TestParticle as Tp


class Interaction:
    def __init__(self):
        # Initialise the system with multiple rings of varying test particle density.
        # Initially, particles will be evenly spaced along the ring

        radii_density = [(2, 12), (3, 18), (4, 24), (5, 30), (6, 36)]
        particles = []

        # Initialisation for circular orbits
        for radii in radii_density:
            radius = radii[0]
            v_theta = np.sqrt(1 / radius ** 2)

            for i in range(radii[1]):
                tp = Tp.TestParticle(position=(radius, i*2*np.pi/radii[1], 0), velocity=(0, v_theta, 0))
                particles.append(tp)


if __name__ == "__main__":
    newInt = Interaction()
