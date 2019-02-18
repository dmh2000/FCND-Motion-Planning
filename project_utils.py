import numpy as np
import numpy.linalg as LA
import re
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from bresenham import bresenham
from queue import PriorityQueue
import networkx as nx
import time


def read_environment(filename="colliders.csv"):
    """
    read the environment data
    :param filename:
    :return: tuple of lat,lon,data
    """
    # read first line and extract lat/lon
    with open(filename) as f:
        line = f.readline()
        # lat0 37.792480, lon0 -122.397450
        m = re.match(r"lat0\s(.*),\slon0 (.*)\s$", line)
        lat = float(m.group(1))
        lon = float(m.group(2))

    # the collider data
    csv_data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)

    # return the file data
    return lat, lon, csv_data


def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)


def collinearity_check(p1, p2, p3, epsilon=1e-6):
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon


def prune_path(path):
    if path is None:
        return path
    if len(path) < 4:
        return path

    pruned_path = []
    p1 = path[0]
    pruned_path.append(p1)
    p2 = path[1]
    for p in path[2:]:
        p3 = p
        if collinearity_check(point(p1), point(p2), point(p3)):
            # skip p1
            p2 = p3
        else:
            pruned_path.append(p2)
            p1 = p2
            p2 = p3

    return pruned_path


# =======================================
# ASTAR GRAPH
# =======================================

def make_node(cost, pos):
    return (cost, pos)


# modify A* to work with a graph
def a_star_graph(graph, h, start, goal):
    """
    find shortest path from start to goal in graph
    Position = tuple(x,y)
    :param graph: networkx undirected graph
    :param h: heuristic function
    :param start: Position
    :param goal: Position
    :return:
    """
    # graph is dictionary with key = Position, value = next position
    # visited is set with key = Position
    # branch is dictionary with key = Position, value = (Position,cost)
    # queue is priority queue with elements = (Position,cost)
    # goal = Position
    # start = Position
    path = []
    path_cost = 0
    queue = PriorityQueue()
    start_node = (0.0, start)
    queue.put(start_node)
    visited = set(start)
    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]

        # set current cost
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        # if close to goal, quit
        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            # for each node adjacent to current_node
            for next_node in graph[current_node]:
                # get the tuple representation
                cost = graph.edges[current_node, next_node]['weight']
                # sum the current path cost
                branch_cost = current_cost + cost
                # add the heuristic cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(n)
            n = branch[n][1]
        path.append(n)
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        return None, None
    return path[::-1], path_cost


def norm_distance(position, goal_position):
    return LA.norm(np.array(goal_position) - np.array(position))


def find_nearest(graph, grid, pos):
    mind = 1000000
    p = None
    for n in graph:
        a = np.array(n)
        if grid[int(n[0]), int(n[1])] == 1:
            continue
        b = pos
        d = LA.norm(a - b)
        if d < mind:
            mind = d
            p = a
    return p[0], p[1]


def create_graph(data, drone_altitude, safety_distance):
    """
    create a grid and graph for astar planning
    :param data:
    :param drone_altitude:
    :param safety_distance:
    :return: populated grid , edges (for display) and voronoi graph though open paths
    """
    grid, edges, north_min, east_min = create_grid_and_edges(data, drone_altitude, safety_distance)
    print('Found %5d edges' % len(edges))

    # =====================================
    # create graph
    # =====================================
    # create the graph with the weight of the edges
    # set to the Euclidean distance between the points
    graph = nx.Graph()

    for e in edges:
        dist = norm_distance(e[0], e[1])  # LA.norm(np.array(e[0]) - np.array(e[1]))
        graph.add_edge(e[0], e[1], weight=dist)

    return graph, edges, grid, north_min, east_min


def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min),
                int(north + d_north + safety_distance - north_min),
                int(east - d_east - safety_distance - east_min),
                int(east + d_east + safety_distance - east_min),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)
    # check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        # integers for bresenham
        p1 = graph.vertices[v[0]].astype(int)
        p2 = graph.vertices[v[1]].astype(int)
        # floats for edges
        b1 = graph.vertices[v[0]]
        b2 = graph.vertices[v[1]]
        # test each pair p1 and p2 for collision using Bresenham
        # If the edge does not hit an obstacle add it to the list
        in_collision = False
        ridgeline = bresenham(p1[0], p1[1], p2[0], p2[1])
        for b in ridgeline:
            # eliminate out of range points in the line
            if b[0] < 0 or b[0] >= grid.shape[0]:
                in_collision = True
                break
            if b[1] < 0 or b[1] >= grid.shape[1]:
                in_collision = True
                break
            # check if grid cell is an obstacle
            if grid[b[0], b[1]] == 1:
                in_collision = True
                break
        # keep ridge points not in collision
        if not in_collision:
            # save floating point values
            b1 = (b1[0], b1[1])
            b2 = (b2[0], b2[1])
            edges.append((b1, b2))

    return grid, edges, north_min, east_min


def plot_grid(prefix, start, goal, grid, path=None, edges=None, waypoints=None):
    # interactive mode off
    plt.ioff()

    # equivalent to
    # plt.imshow(np.flip(grid, 0))
    # Plot it up!
    plt.imshow(grid, origin='lower', cmap='Greys')

    # plot edges if any
    if edges is not None:
        for e in edges:
            p1 = e[0]
            p2 = e[1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

    # plot path if any
    if path is not None:
        p1 = path[0]
        for p in path[1:]:
            p2 = p
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')
            p1 = p2

    if waypoints is not None:
        for wp in waypoints:
            plt.plot(wp[0], wp[1], 'go')

    plt.plot(start[1], start[0], 'rx')
    plt.plot(goal[1], goal[0], 'rx')
    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    t = time.strftime("%m%d%H%M%S")
    fname = "{0}_{1}-{2}-{3}-{4}-{5}.jpg".format(prefix, int(start[0]), int(start[1]), int(goal[0]), int(goal[1]), t)
    plt.savefig(fname)


# test
if __name__ == "__main__":
    lat, lon, data = read_environment()
    print(lat, lon, len(data))
