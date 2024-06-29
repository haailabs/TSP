# TSP Solver with Lin-Kernighan-Helsgaun (LKH) Heuristic

This project implements a Traveling Salesman Problem (TSP) solver using the Lin-Kernighan-Helsgaun (LKH) heuristic, along with various TSP instance generators and visualization tools.

## Files

1. `LKH.py`: Contains the implementation of the LKH algorithm.
2. `main.py`: Includes TSP generators, visualization tools, and example usage of the LKH solver.

## LKH.py

This file implements the Lin-Kernighan-Helsgaun heuristic for solving the Traveling Salesman Problem. Key features include:

- Candidate set generation
- Alpha-nearness calculation
- Double-bridge kick for escaping local optima
- Iterative improvement with a don't look bit optimization

### Class: LKH

Main methods:
- `__init__(self, distances)`: Initialize the LKH solver with a distance matrix
- `run(self, max_iterations, max_no_improve)`: Run the LKH algorithm
- Various helper methods for move evaluation and tour improvement

## main.py

This file provides tools for generating TSP instances, visualizing solutions, and demonstrating the use of the LKH solver. Key components include:

### Class: TSPGenerator

Methods for generating various types of TSP instances:
- `generate_symmetric_tsp()`
- `generate_asymmetric_tsp()`
- `generate_cvrp()`
- `generate_euclidean_tsp()`
- `generate_hamiltonian_cycle_problem()`
- `generate_geographic_tsp()`

### Visualization

- `visualize_geographic_tsp()`: Create a map visualization of a geographic TSP instance
- `visualize_tsp_solution()`: Visualize a TSP solution on a map

### LKH Integration

- `run_lkh()`: Solve a TSP instance using the LKH algorithm

## Dependencies

- random
- math
- typing
- numpy
- osmnx
- networkx
- folium
- geopy
