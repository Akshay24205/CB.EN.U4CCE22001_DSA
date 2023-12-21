import heapq

def dijkstra(graph, source):
    # Initialize distances with a large value for all vertices except the source
    distances = {vertex: float('inf') for vertex in graph}
    distances[source] = 0  # Distance from source to itself is 0

    # Priority queue to store vertices and their current distances
    priority_queue = [(0, source)]  # (distance, vertex)
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # Skip if the distance to the current vertex is greater than the known distance
        if current_distance > distances[current_vertex]:
            continue

        # Explore neighbors and update distances
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            # Update distance if a shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# Example graph (adjacency list representation)
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'D': 5},
    'C': {'A': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

source_vertex = 'A'  # Define the source vertex

shortest_distances = dijkstra(graph, source_vertex)
print(f"Shortest distances from '{source_vertex}' to all other vertices:")
for vertex, distance in shortest_distances.items():
    print(f"To reach vertex '{vertex}' - Shortest distance: {distance}")


class Graph:
    def __init__(self, vertices):
        self.V = vertices  # Number of vertices
        self.graph = []    # List to store graph edges

    def add_edge(self, u, v, w):
        # Add an edge: u -> v with weight w
        self.graph.append([u, v, w])

    def bellman_ford(self, src):
        # Initialize distances from the source vertex to all other vertices as infinity
        distance = [float('inf')] * self.V
        distance[src] = 0  # Distance from source to itself is 0

        # Find shortest path for all vertices
        for _ in range(self.V - 1):
            # Iterate through all edges (u, v, w)
            for u, v, w in self.graph:
                if distance[u] != float('inf') and distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w

        # Check for negative cycles by iterating through all edges again
        for u, v, w in self.graph:
            if distance[u] != float('inf') and distance[u] + w < distance[v]:
                print("Graph contains negative cycle")
                return

        # Print the shortest distances from the source vertex
        print("Shortest distances from the source vertex:")
        for i in range(self.V):
            print(f"To reach vertex {i} - Shortest distance: {distance[i]}")

# Example usage:
g = Graph(5)  # Create a graph with 5 vertices
g.add_edge(0, 1, 4)
g.add_edge(0, 2, 5)
g.add_edge(1, 2, -2)
g.add_edge(1, 3, 6)
g.add_edge(2, 3, 1)
g.add_edge(2, 4, 3)
g.add_edge(3, 4, 2)

source_vertex = 0  # Define the source vertex

g.bellman_ford(source_vertex)


INF = float('inf')

def floyd_warshall(graph):
    v = len(graph)
    
    # Create a copy of the graph as the distance matrix
    distance = [row[:] for row in graph]

    # Iterate through all vertices 'k'
    for k in range(v):
        # For each pair of vertices 'i' and 'j', update the distance if the path through 'k' is shorter
        for i in range(v):
            for j in range(v):
                if distance[i][k] != INF and distance[k][j] != INF and distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]

    return distance

# Example graph represented as an adjacency matrix
# Replace INF with float('inf') for infinity in the graph
graph = [
    [0, 5, INF, 10],
    [INF, 0, 3, INF],
    [INF, INF, 0, 1],
    [INF, INF, INF, 0]
]

shortest_distances = floyd_warshall(graph)

# Display the shortest distances between all pairs of vertices
print("Shortest distances between all pairs of vertices:")
for row in shortest_distances:
    print(row)


from collections import defaultdict, deque

def bfs(graph, start_node):
    visited = set()  # Initialize a set to keep track of visited vertices
    queue = deque([start_node])  # Initialize a queue for BFS traversal
    visited.add(start_node)  # Mark the start node as visited

    while queue:
        current_node = queue.popleft()
        print(current_node)  # Print the current node during traversal

        # Enqueue adjacent vertices of the current node
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)  # Mark neighbor as visited

# Example graph represented as an adjacency list
# Format: {vertex: [list of adjacent vertices]}
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Perform BFS traversal starting from node 'A'
print("BFS traversal starting from node 'A':")
bfs(graph, 'A')


def dfs(graph, start_node, visited=None):
    if visited is None:
        visited = set()  # Initialize a set to keep track of visited vertices

    print(start_node)  # Print the current node during traversal
    visited.add(start_node)  # Mark the current node as visited

    # Explore adjacent vertices of the current node recursively
    for neighbor in graph[start_node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Example graph represented as an adjacency list
# Format: {vertex: [list of adjacent vertices]}
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Perform DFS traversal starting from node 'A'
print("DFS traversal starting from node 'A':")
dfs(graph, 'A')


INF = float('inf')

def print_mst(parent, graph):
    print("Edge \tWeight")
    for i in range(1, len(graph)):
        print(f"{parent[i]} - {i}\t{graph[i][parent[i]]}")

def prim_mst(graph):
    V = len(graph)  # Number of vertices in the graph

    # Initialize lists to store MST and keys (minimum edge weights)
    parent = [-1] * V  # To store constructed MST
    key = [INF] * V     # Key values used to pick minimum weight edge

    # Mark the first vertex as part of MST
    key[0] = 0
    parent[0] = -1

    for _ in range(V - 1):
        # Find the vertex with the minimum key value
        min_key = INF
        min_index = -1
        for v in range(V):
            if key[v] < min_key and v not in parent:
                min_key = key[v]
                min_index = v

        u = min_index

        # Update key and parent for adjacent vertices of the selected vertex
        for v in range(V):
            if 0 < graph[u][v] < key[v] and v not in parent:
                key[v] = graph[u][v]
                parent[v] = u

    print_mst(parent, graph)

# Example graph represented as an adjacency matrix
# Replace INF with float('inf') for infinity in the graph
graph = [
    [0, 2, 0, 6, 0],
    [2, 0, 3, 8, 5],
    [0, 3, 0, 0, 7],
    [6, 8, 0, 0, 9],
    [0, 5, 7, 9, 0]
]

# Find and print the Minimum Spanning Tree (MST) using Prim's algorithm
prim_mst(graph)


class DisjointSet:
    def __init__(self, vertices):
        self.parent = {vertex: vertex for vertex in vertices}
        self.rank = {vertex: 0 for vertex in vertices}

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, vertex1, vertex2):
        root1 = self.find(vertex1)
        root2 = self.find(vertex2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def kruskal_mst(graph):
    vertices = list(graph.keys())
    edges = []
    for vertex in vertices:
        for neighbor, weight in graph[vertex]:
            edges.append((weight, vertex, neighbor))

    edges.sort()  # Sort edges in ascending order by weight
    mst = []
    disjoint_set = DisjointSet(vertices)

    for edge in edges:
        weight, vertex1, vertex2 = edge
        if disjoint_set.find(vertex1) != disjoint_set.find(vertex2):
            mst.append((vertex1, vertex2, weight))
            disjoint_set.union(vertex1, vertex2)

    return mst

# Example graph represented as an adjacency list of edges with weights
graph = {
    'A': [('B', 2), ('C', 3)],
    'B': [('A', 2), ('D', 5), ('C', 6)],
    'C': [('A', 3), ('B', 6), ('D', 4)],
    'D': [('B', 5), ('C', 4)]
}

# Find and print the Minimum Spanning Tree (MST) using Kruskal's algorithm
minimum_spanning_tree = kruskal_mst(graph)
print("Minimum Spanning Tree (MST):")
for edge in minimum_spanning_tree:
    print(f"{edge[0]} - {edge[1]}: {edge[2]}")



