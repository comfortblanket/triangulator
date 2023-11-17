# Triangulator

This is currently a first draft of an algorithm to estimate the source of a signal when it is detected at various physically-distributed sensors with different times of arrival (TOAs).

The algorithm currently simply loops over a search area and calculates a simplistic squared-differences score function for each location. This can then be plotted on top of a map to see possible source locations.

The algorithm needs to be updated to a statistical approach, but for now it's a cute little toy.