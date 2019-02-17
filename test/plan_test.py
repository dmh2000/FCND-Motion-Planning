import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid
from project_utils import read_environment, prune_path, create_graph, a_star_graph, find_nearest, plot_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 5


def plan_path(start, goal):
    print("Searching for a path ...")

    # TODO: read lat0, lon0 from colliders into floating point values
    lat0, lon0, data = read_environment()

    # Define a grid for a particular altitude and safety margin around obstacles
    graph, edges, grid = create_graph(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
    print("nodes : {0} edges = {1} graph = {2}".format(len(list(graph)), len(edges), grid.shape))

    # Define starting point on the grid (this is just grid center)
    # TODO: convert start position to current position rather than map center
    # find nearest points in the graph
    start = find_nearest(graph, grid, start)
    goal = find_nearest(graph, grid, goal)

    # PRINT STRAIGHT LINE DISTANCE START TO GOAL
    print("START->GOAL :", heuristic(start, goal))


    # Run A* to find a path from start to goal
    # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
    # or move to a different search space such as a graph (not done here)
    print('Local Start and Goal: ', start, goal)
    path, _ = a_star_graph(graph, heuristic, start, goal)

    # TODO: prune path to minimize number of waypoints
    # TODO (if you're feeling ambitious): Try a different approach altogether!
    # path = prune_path(grid, path)

    plot_grid(start, goal, grid, path, edges)

    # Convert path to waypoints
    waypoints = [[p[0], p[1], TARGET_ALTITUDE, 0] for p in path]


if __name__ == "__main__":
    start = (25, 100)
    goal = (750, 370)

    start = (475, 460)
    goal = (870, 360)
    plan_path(start, goal)
