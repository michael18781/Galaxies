# Computational Physics Project on interacting galaxies

## Stage 1
Built a program to compute the motion in circular orbits around a heavy mass.

## Stage 2
Modified the program to allow for multiple heavy masses (though they must remain static) and arbitrary orbits. 
This can be found as the Interaction class. It is tested for particles at radii 2, 3, 4, 5, 6 with numbers 12, 18, 24, 30, 36
and produces the following plot:

![](/images/circularOrbitsInteraction.png)

Two masses of mass = 1 at (0, 0) and mass = 3 at (4, 0) and a particle of initial velocity
of 1 unit we see precessing orbits:

![](/images/ellipticalOrbitsInteraction.png)

One mass of mass = 1 at (0, 0) and a particle of initial velocity 0.2 units we see an elliptical orbit:

![](/images/1_unit_at_origin_v=0.2.png)

## Stage 3
Extended the program to include two heavy masses which interact with each other and any number of test masses
which feel forces from the two heavy masses. This can be found as the MultipleParticleSystem class.

The MultipleParticleSystem is an improved more general version of the Interaction class and so to continue on using it,
we should check it agrees on edge cases e.g. the two masses solution with precessing elliptical solutions and the simple
circular orbits. It does agree.
