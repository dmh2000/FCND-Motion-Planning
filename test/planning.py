import numpy as np
from udacidrone.frame_utils import global_to_local, local_to_global
from project_utils import read_environment, prune_path
from project_utils import create_graph, a_star_graph
from project_utils import find_nearest, plot_grid, norm_distance


def plan_path():
    print("Searching for a path ...")
    TARGET_ALTITUDE = 5
    SAFETY_DISTANCE = 5

    # set takeoff altitude
    target_position = [0, 0, 0]
    target_position[2] = TARGET_ALTITUDE

    # TODO: read lat0, lon0 from colliders into floating point values
    lat0, lon0, data = read_environment("../colliders.csv")

    # TODO: set home position to (lon0, lat0, 0)
    # set_home_position(lon0, lat0, 0)

    # TODO: retrieve current global position
    global_position = np.array([-122.397450, 37.7924794, 0])
    global_goal = np.array([-122.3987356, 37.7961938, 0])
    local_position = [-0.06026223, -0.03686861, -0.01355824]

    # TODO: convert to current local position using global_to_local()
    global_home = global_position
    current_position = global_to_local(global_position, global_home)

    # print('global home {0}, global_position {1}, local position {2}, current_position {3}'.format(global_home,
    #                                                                                              global_position,
    #                                                                                              local_position,
    #                                                                                              current_position))

    # Define a grid for a particular altitude and safety margin around obstacles
    graph, edges, grid, north_min, east_min = create_graph(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
    # print("nodes : {0} edges = {1} graph = {2}".format(len(list(graph)), len(edges), grid.shape))

    # Define starting point on the grid (this is just grid center)

    # TODO: convert start position to current position rather than map center
    # get start in grid coordinates
    start_p = (current_position[0] - north_min + 100, current_position[1] - east_min)

    # Set goal as some arbitrary position on the grid
    # TODO: adapt to set goal as latitude / longitude position and convert
    local_goal = global_to_local(global_goal, global_home)
    # get goal in grid coordinates
    goal_p = (local_goal[0] - north_min, local_goal[1] - east_min)

    # find nearest points in the graph
    start_p = find_nearest(graph, grid, start_p)
    goal_p = find_nearest(graph, grid, goal_p)

    # Run A* to find a path from start to goal
    # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
    # or move to a different search space such as a graph (not done here)
    print('Local Start and Goal: ', start_p, goal_p)
    # path, _ = a_star_graph(graph, norm_distance, start_p, goal_p)
    path, _ = a_star_graph(graph, norm_distance, start_p, goal_p)


    # TODO: prune path to minimize number of waypoints
    # TODO (if you're feeling ambitious): Try a different approach altogether!
    # path = prune_path(grid, path)

    # Convert path to waypoints
    # waypoints = [[int(p[0]) + north_min, int(p[1]) + east_min, TARGET_ALTITUDE, 0] for p in path]

    plot_grid(start_p,
              goal_p,
              grid,
              path,
              edges)

if __name__ == "__main__":
    plan_path()
