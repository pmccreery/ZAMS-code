# Zero-Age Main Sequence Stellar Structure

This repository houses a very basic ZAMS stellar structure model for graduate level stellar structure course. We have numerically solved the four, coupled, non-linear differential equations that describes stellar structure using the shooting method and the Newton-Raphson method.

Quick outline of the repository:

Report/ - contains files to compile the report on this project
opacities/ - contains the files used to create the stitched opacity table

calculations.py - calculations of various sorts (energy generation, derivatives, etc.)

constants.py - contains cgs constants needed (courtesy of Kevin Schlaufman)

mesa2sol.txt - contains the output for the MESA run of a 2 solar mass star

newton.py - Newton-Raphson method

properties.py - properties of the star (THE STARTING POINT)

resulttolatex.py - takes the results of the shooting method and turns it into a machine readable table

run.py - runs the whole code

shootf.py - contains the functions relating to the shooting method

stellarCode.ipynb - Mainly a test ground for the code -- similar to run.py

stellarStructure.yml - Conda environment to run this code with

tableInterpolation.py - interpolation routine, table stitching
