import random
import math
import osmnx as ox
import networkx as nx
import folium
from folium import FeatureGroup, LayerControl
from geopy.geocoders import Nominatim
from LKH import LKH


class TSPGenerator:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 10000)
        random.seed(self.seed)

    def generate_symmetric_tsp(self, num_nodes, min_distance=1, max_distance=100):
        distances = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = random.randint(min_distance, max_distance)
                distances[i][j] = distance
                distances[j][i] = distance
        return distances

    def generate_asymmetric_tsp(self, num_nodes, min_distance=1, max_distance=100):
        distances = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    distances[i][j] = random.randint(min_distance, max_distance)
        return distances

    def generate_cvrp(self, num_nodes, min_demand=1, max_demand=10, truck_capacity=50, min_distance=1, max_distance=100):
        distances = self.generate_symmetric_tsp(num_nodes + 1, min_distance, max_distance)
        demands = [random.randint(min_demand, max_demand) for _ in range(num_nodes)]
        return distances, demands, truck_capacity

    def generate_euclidean_tsp(self, num_nodes, min_coord=0, max_coord=100):
        nodes = [(random.uniform(min_coord, max_coord), random.uniform(min_coord, max_coord)) for _ in range(num_nodes)]
        distances = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = math.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)
                distances[i][j] = distance
                distances[j][i] = distance
        return distances, nodes

    def generate_hamiltonian_cycle_problem(self, num_nodes, edge_probability=0.5):
        graph = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_probability:
                    graph[i][j] = 1
                    graph[j][i] = 1
        return graph

    def get_seed(self):
        return self.seed


    def generate_geographic_tsp(self, location, num_nodes, distance=1000):
        """
        Generate a geographic TSP based on real map data.

        :param location: Name of the location (e.g., "New York, NY")
        :param num_nodes: Number of nodes to generate
        :param distance: Approximate radius of the area to consider (in meters)
        :return: Dictionary with nodes, distances, and graph
        """
        # Fetch the street network
        G = ox.graph_from_address(location, dist=distance, network_type='drive')

        # Get a list of node coordinates
        node_coords = [(data['y'], data['x']) for node, data in G.nodes(data=True)]

        # Randomly select nodes
        selected_nodes = random.sample(node_coords, num_nodes)

        # Calculate distances between nodes
        distances = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                try:
                    nearest_node_i = ox.distance.nearest_nodes(G, selected_nodes[i][1], selected_nodes[i][0])
                    nearest_node_j = ox.distance.nearest_nodes(G, selected_nodes[j][1], selected_nodes[j][0])
                    route = nx.shortest_path(G, nearest_node_i, nearest_node_j, weight='length')
                    distance = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
                    distances[i][j] = distance
                    distances[j][i] = distance
                except nx.NetworkXNoPath:
                    # If no path exists, set a very large distance
                    distances[i][j] = float('inf')
                    distances[j][i] = float('inf')

        return {
            'nodes': selected_nodes,
            'distances': distances,
            'graph': G
        }
    def run_lkh(self, distances):
        """
        Solve TSP using the LKH algorithm.

        :param distances: 2D list of distances between nodes
        :return: List representing the tour
        """
        lkh = LKH(distances)
        best_tour, best_distance = lkh.run(max_iterations=1000, max_no_improve=100)
        return best_tour
    def visualize_geographic_tsp(self, tsp_data):
        """
        Visualize the geographic TSP on a map with a random solution.

        :param tsp_data: Dictionary returned by generate_geographic_tsp
        :return: Folium map object
        """
        G = tsp_data['graph']
        nodes = tsp_data['nodes']
        distances = tsp_data['distances']

        # Create a random solution
        solution = list(range(len(nodes)))
        random.shuffle(solution)
        solution.append(solution[0])  # Return to start

        # Create a map centered on the mean coordinates of the nodes
        center_lat = sum(node[0] for node in nodes) / len(nodes)
        center_lon = sum(node[1] for node in nodes) / len(nodes)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Create a feature group for the route
        route_group = FeatureGroup(name="Route")

        # Add markers for each node
        for i, (lat, lon) in enumerate(nodes):
            folium.Marker(
                [lat, lon],
                popup=f'Node {i}',
                tooltip=f'({lat:.6f}, {lon:.6f})'
            ).add_to(m)

        # Draw the route and calculate total distance
        total_distance = 0
        for i in range(len(solution) - 1):
            start = solution[i]
            end = solution[i + 1]
            start_node = nodes[start]
            end_node = nodes[end]

            # Find the nearest network nodes
            start_nearest = ox.distance.nearest_nodes(G, start_node[1], start_node[0])
            end_nearest = ox.distance.nearest_nodes(G, end_node[1], end_node[0])

            # Get the shortest path
            try:
                route = nx.shortest_path(G, start_nearest, end_nearest, weight='length')
                edge_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

                # Calculate the distance for this edge
                route_gdf = ox.routing.route_to_gdf(G, route)
                edge_distance = route_gdf['length'].sum()
                total_distance += edge_distance

                # Draw the route
                folium.PolyLine(
                    edge_coords,
                    color="red",
                    weight=2.5,
                    opacity=0.8,
                    tooltip=f'Distance: {edge_distance:.2f} meters'
                ).add_to(route_group)

            except nx.NetworkXNoPath:
                print(f"No path found between nodes {start} and {end}")

        route_group.add_to(m)

        # Add layer control
        LayerControl().add_to(m)

        # Add total distance information
        folium.Rectangle(
            bounds=[[0, 0], [1, 1]],
            color="white",
            fill=True,
            popup=f'Total Distance: {total_distance:.2f} meters'
        ).add_to(m)

        return m
    def visualize_tsp_solution(self, tsp_data, solution):
        """
        Visualize a TSP solution on a map.

        :param tsp_data: Dictionary returned by generate_geographic_tsp
        :param solution: List representing the tour
        :return: Folium map object
        """
        G = tsp_data['graph']
        nodes = tsp_data['nodes']
        distances = tsp_data['distances']

        # Create a map centered on the mean coordinates of the nodes
        center_lat = sum(node[0] for node in nodes) / len(nodes)
        center_lon = sum(node[1] for node in nodes) / len(nodes)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Create a feature group for the route
        route_group = FeatureGroup(name="Route")

        # Add markers for each node
        for i, (lat, lon) in enumerate(nodes):
            folium.Marker(
                [lat, lon],
                popup=f'Node {i}',
                tooltip=f'({lat:.6f}, {lon:.6f})'
            ).add_to(m)

        # Draw the route and calculate total distance
        total_distance = 0
        for i in range(len(solution) - 1):
            start = solution[i]
            end = solution[i + 1]
            start_node = nodes[start]
            end_node = nodes[end]

            # Find the nearest network nodes
            start_nearest = ox.distance.nearest_nodes(G,       start_node[1], start_node[0])
            end_nearest = ox.distance.nearest_nodes(G, end_node[1],  end_node[0])

            # Get the shortest path
            try:
                route = nx.shortest_path(G, start_nearest, end_nearest, weight='length')
                edge_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

                # Calculate the distance for this edge
                route_gdf = ox.routing.route_to_gdf(G, route)
                edge_distance = route_gdf['length'].sum()
                total_distance += edge_distance

                # Draw the route
                folium.PolyLine(
                    edge_coords,
                    color="red",
                    weight=2.5,
                    opacity=0.8,
                    tooltip=f'Distance: {edge_distance:.2f} meters'
                ).add_to(route_group)

            except nx.NetworkXNoPath:
                print(f"No path found between nodes {start} and {end}")

        route_group.add_to(m)

        # Add layer control
        LayerControl().add_to(m)

        # Add total distance information
        folium.Rectangle(
            bounds=[[0, 0], [1, 1]],
            color="white",
            fill=True,
            popup=f'Total Distance: {total_distance:.2f} meters'
        ).add_to(m)

        return m, total_distance
# Example usage
generator = TSPGenerator(seed=42)

# Generate Geographic TSP
geo_tsp = generator.generate_geographic_tsp("Central Park, New York", 10, distance=2000)
print("\nGeographic TSP:")
print("Nodes:")
for i, node in enumerate(geo_tsp['nodes']):
    print(f"Node {i}: {node}")
print("\nDistances:")
for row in geo_tsp['distances']:
    print(row)

# Generate a solution using LKH
lkh_solution = generator.run_lkh(geo_tsp['distances'])
print("\nLKH Solution:", lkh_solution)

# Visualize the LKH solution
map_vis, total_distance = generator.visualize_tsp_solution(geo_tsp, lkh_solution)


map_vis.save("geographic_tsp_map_lkh_solution.html")
print(f"\nMap visualization with LKH solution saved as 'geographic_tsp_map_lkh_solution.html'")
print(f"Total distance of LKH solution: {total_distance:.2f} meters")

print("\nSeed used:", generator.get_seed())