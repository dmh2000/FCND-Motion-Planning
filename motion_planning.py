import argparse
import time
import msgpack
from enum import Enum, auto
import numpy as np
import pickle

from planning_utils import a_star, heuristic, create_grid
from project_utils import read_environment, prune_path, pickle_log
from project_utils import create_graph, a_star_graph, find_nearest, plot_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5
    PLANNING = 6


class Events(Enum):
    LOCAL_POSITION = 0
    VELOCITY = 1
    STATE = 2


class Planner(Enum):
    GRID_SEARCH = 0
    VORONOI = 1


# NED COORDINATE SYSTEM
# right handed (down is positive)
# X is positive north (up in simulator)
# Y is positive east  (right in simulator)
# Z is psitive down


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 5.0])
        self.in_mission = True
        self.check_state = {}

        # set waypoint parameters
        self.waypoints = []  # list of waypoints in flight order

        # plan data
        self.enable_plot = False
        self.global_goal = [-122.397896, 37.792523, 0]
        self.planner = Planner.GRID_SEARCH  # default to grid search
        self.waypoint_tolerance = 8.0
        self.loiter_seconds = 4
        self.loiter_state = 0
        self.loiter_t0 = 0.0

        # initial state
        self.flight_state = States.MANUAL

        # register callbacks
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    # ===============================================================
    # callbacks dispatch event to state handler
    # Event handling was refactored to 'state_handler' below
    # ===============================================================

    def local_position_callback(self):
        """
        This triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data
        """
        # process the event
        self.state_handler(Events.LOCAL_POSITION)

    def velocity_callback(self):
        """
        This triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data
        """
        # process the event
        self.state_handler(Events.VELOCITY)

    def state_callback(self):
        """
        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        # process the event
        self.state_handler(Events.STATE)

    # ===============================================================
    # state transitions
    # ===============================================================

    def arming_transition(self):
        """
        arm and take control
        """
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        """
        command takeoff
        """
        self.flight_state = States.TAKEOFF
        print("takeoff transition", self.target_position)
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        """
        1. Command the next waypoint position
        2. Transition to WAYPOINT state
        """
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0],
                          self.target_position[1],
                          self.target_position[2],
                          self.target_position[3])
        # reset loiter state for this waypoint
        self.loiter_state = 0

    def landing_transition(self):
        """T
        1. Command the drone to land
        2. Transition to the LANDING state
        """
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        """
        1. Command the drone to disarm
        2. Transition to the DISARMING state
        """
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        """
        1. Write a pickle of the telemetry log for later plotting
        2. Release control of the drone
        3. Stop the connection (and telemetry log)
        4. End the mission
        5. Transition to the MANUAL state
        """
        print("manual transition")
        pickle_log(self)
        self.stop()
        self.in_mission = False

    # ===============================================================
    # state handlers
    # ===============================================================

    def manual_state(self, event):
        """
        handle events when in this state
        :param event: position,velocity or state
        :return: none
        """
        if event == Events.LOCAL_POSITION:
            pass
        elif event == Events.VELOCITY:
            pass
        elif event == Events.STATE:
            # starting up, initiate arming
            self.arming_transition()
        else:
            print("INVALID EVENT", self.flight_state, event)

    def arming_state(self, event):
        """
        handle events when in this state
        :param event: position,velocity or state
        :return: none
        """
        if event == Events.LOCAL_POSITION:
            pass
        elif event == Events.VELOCITY:
            pass
        elif event == Events.STATE:
            # wait until armed then initiate planning
            if self.armed and self.guided:
                if self.planner == Planner.VORONOI:
                    self.plan_path_voronoi()
                else:
                    self.plan_path_grid()
        else:
            print("INVALID EVENT", self.flight_state, event)

    def planning_state(self, event):
        """
        handle events in this state
        :param event: position,velocity or state
        :return:
        """
        if event == Events.LOCAL_POSITION:
            pass
        elif event == Events.VELOCITY:
            pass
        elif event == Events.STATE:
            # takeoff
            self.takeoff_transition()
        else:
            print("INVALID EVENT", self.flight_state, event)

    def takeoff_state(self, event):
        """
        handle events when in this state
        :param event: position,velocity or state
        :return: none
        """
        if event == Events.LOCAL_POSITION:
            # check position and go to first waypoint if the position is in bounds
            # coordinate conversion
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif event == Events.VELOCITY:
            pass
        elif event == Events.STATE:
            pass
        else:
            print("INVALID EVENT", self.flight_state, event)

    def waypoint_state(self, event):
        """
        handle events when in this state
        :param event: position,velocity or state
        :return: none
        """
        # NEW =====================================================
        # increased tolerance for stepping waypoint
        # NEW =====================================================
        if event == Events.LOCAL_POSITION:
            # check waypoint bounds and go to next
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < self.waypoint_tolerance:
                if len(self.waypoints) > 0:
                    # NEW =====================================================
                    # LOITER OPTION
                    # NEW =====================================================
                    if self.loiter_seconds == 0:
                        # no loiter at waypoint
                        self.waypoint_transition()
                    else:
                        if self.loiter_state == 0:
                            self.loiter_t0 = time.monotonic()
                            self.loiter_state = 1
                        elif self.loiter_state == 1:
                            t = time.monotonic()
                            # wait for loiter time to expire
                            if (t - self.loiter_t0) >= self.loiter_seconds:
                                # time expired, next waypoint
                                self.loiter_state = 0
                                self.waypoint_transition()
                        else:
                            print("Invalid loiter state")
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

        elif event == Events.VELOCITY:
            pass
        elif event == Events.STATE:
            pass
        else:
            print("INVALID EVENT", self.flight_state, event)

    def landing_state(self, event):
        """
        handle events when in this state
        :param event: position,velocity or state
        :return: none
        """
        if event == Events.LOCAL_POSITION:
            # check position and go to disarming if in touchdown bounds
            # why was this in the velocity callback in the up-down exaple?
            if ((self.global_position[2] - self.global_home[2] < 0.1) and
                    abs(self.local_position[2]) < 0.01):
                self.disarming_transition()
        elif event == Events.VELOCITY:
            # check position and go to disarming if in touchdown bounds
            # why was this in the velocity callback in the up-down exaple?
            if ((self.global_position[2] - self.global_home[2] < 0.1) and
                    abs(self.local_position[2]) < 0.01):
                self.disarming_transition()
            pass
        elif event == Events.STATE:
            pass
        else:
            print("INVALID EVENT", self.flight_state, event)

    def disarming_state(self, event):
        """
        handle events when in this state
        :param event: position,velocity or state
        :return: none
        """
        if event == Events.LOCAL_POSITION:
            pass
        elif event == Events.VELOCITY:
            pass
        elif event == Events.STATE:
            # shutdown
            if ~self.armed & ~self.guided:
                self.manual_transition()
        else:
            print("INVALID EVENT", self.flight_state, event)

    def state_handler(self, event):
        """
        all events come here first and are then
        dispatched to current state handler
        :param event:
        :return: none
        """
        # get current state
        state = self.flight_state

        # process the event for the current state
        if state == States.MANUAL:
            self.manual_state(event)
        elif state == States.ARMING:
            self.arming_state(event)
        elif state == States.PLANNING:
            self.planning_state(event)
        elif state == States.TAKEOFF:
            self.takeoff_state(event)
        elif state == States.WAYPOINT:
            self.waypoint_state(event)
        elif state == States.LANDING:
            self.landing_state(event)
        elif state == States.DISARMING:
            self.disarming_state(event)
        else:
            print("INVALID STATE")
            exit(1)

    def send_waypoints(self):
        """
        send waypoints to simulator for visualization
        """
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def set_goal(self, lat, lon):
        self.global_goal = [lon, lat, 0]

    def plan_path_grid(self):
        """
        plan the path using grid search
        """
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 8

        # compute elapsed time to plan
        t0 = time.monotonic()

        # set takeoff altitude
        self.target_position[2] = TARGET_ALTITUDE

        # read lat0, lon0 from colliders into floating point values
        lat0, lon0, data = read_environment()

        # set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # retrieve current global position
        global_position = self.global_position

        # convert to current local position using global_to_local()
        current_position = global_to_local(global_position, self.global_home)

        # set initial location and takeoff altitude
        # self.target_position = current_position;
        self.target_position[2] = TARGET_ALTITUDE

        print('global home {0}\nglobal_position {1}\nlocal position {2}\ncurrent_position {3}'.format(
            self.global_home,
            self.global_position,
            self.local_position,
            current_position)
        )

        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # Define starting point on the grid (this is just grid center)
        # convert start position to current position rather than map center
        grid_start = (int(current_position[0] - north_offset), int(current_position[1] - east_offset))

        # Set goal as some arbitrary position on the grid
        # adapt to set goal as latitude / longitude position and convert
        local_goal = global_to_local(self.global_goal, self.global_home)

        grid_goal = (int(local_goal[0] - north_offset), int(local_goal[1] - east_offset))

        # Run A* to find a path from start to goal
        # see planning_utils.py: add diagonal motions with a cost of sqrt(2) to your A* implementation
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, cost = a_star(grid, heuristic, grid_start, grid_goal)

        # quit if path not found
        if path is None:
            self.disarming_transition()
            return

        # prune path to minimize number of waypoints
        pruned_path = prune_path(path)
        if len(pruned_path) == 0:
            print("PRUNED PATH FAILED")
            pruned_path = path

        print("Path {0}:{1:f} : Pruned Path {2} ".format(len(path), cost, len(pruned_path)))

        # Convert path to waypoints without heading
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in pruned_path]

        # Set self.waypoints
        self.waypoints = waypoints

        if self.enable_plot:
            plot_grid("grid",
                      grid_start,
                      grid_goal,
                      grid,
                      pruned_path)

        # send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

        # print elapsed time
        print("ET: {0:f}".format(time.monotonic() - t0))

        # transition to planning
        self.flight_state = States.PLANNING

    def plan_path_voronoi(self):
        """
        plan the path using a Voronoi graph
        """
        # transition to planning
        self.flight_state = States.PLANNING

        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        # compute elapsed time to plan
        t0 = time.monotonic()

        # set takeoff altitude
        self.target_position[2] = TARGET_ALTITUDE

        # read lat0, lon0 from colliders into floating point values
        lat0, lon0, data = read_environment()

        # set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # retrieve current global position
        global_position = self.global_position

        # convert to current local position using global_to_local()
        current_position = global_to_local(global_position, self.global_home)

        print('global home {0}\nglobal_position {1}\nlocal position {2}\ncurrent_position {3}'.format(
            self.global_home,
            self.global_position,
            self.local_position,
            current_position)
        )

        # Define a graph and grid for a particular altitude and safety margin around obstacles
        # see project_utils.py
        graph, edges, grid, north_offset, east_offset = create_graph(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        # Define starting point on the grid (this is just grid center)
        # convert start position to current position rather than map center
        # get start in grid coordinates
        start_p = (current_position[0] - north_offset, current_position[1] - east_offset)

        # Set goal as some arbitrary position on the grid
        # adapt to set goal as latitude / longitude position and convert
        local_goal = global_to_local(self.global_goal, self.global_home)

        # get goal in grid coordinates
        goal_p = (local_goal[0] - north_offset, local_goal[1] - east_offset)

        # find nearest points in the graph
        print(start_p, goal_p)
        start_p = find_nearest(graph, grid, start_p)
        goal_p = find_nearest(graph, grid, goal_p)

        # Run Graph A* to find a path from start to goal
        # see project_utils.py
        print('Local Start and Goal: {0} {1} '.format(start_p, goal_p))
        path, cost = a_star_graph(graph, heuristic, start_p, goal_p)

        # quit if path not found
        if path is None:
            self.disarming_transition()
            return

        # prune path to minimize number of waypoints
        # prune path to minimize number of waypoints
        pruned_path = prune_path(path)
        if len(pruned_path) == 0:
            print("PRUNED PATH FAILED")
            pruned_path = path

        print("Path {0}:{1:f} : Pruned Path {2} ".format(len(path), cost, len(pruned_path)))

        # waypoints without heading
        # waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in path]

        # Convert path to waypoints with heading
        waypoints = []
        wp1 = [int(pruned_path[0][0] + north_offset), int(pruned_path[0][1] + east_offset), TARGET_ALTITUDE, 0]
        waypoints.append(wp1)
        for p in pruned_path[1:]:
            wp2 = [int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0]
            wp2[3] = np.arctan2(wp2[1] - wp1[1], wp2[0] - wp1[0])
            waypoints.append(wp2)
            wp1 = wp2

        self.waypoints = waypoints

        if self.enable_plot:
            plot_grid("graph",
                      start_p,
                      goal_p,
                      grid,
                      path,
                      edges)

        # send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

        # print elapsed time
        print("ET: {0:f}".format(time.monotonic() - t0))


    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--mode', type=str, default='grid', help="'grid' or'voronoi'")
    parser.add_argument('--plot', action='store_true', help="enables plot of grid,path and graph (if any)")
    args = parser.parse_args()

    # conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=10)
    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=1200)
    drone = MotionPlanning(conn)

    # set to true to generate a plot of the grid, path and graph (if using voronoi)
    if args.plot:
        print("plot enabled")
    drone.enable_plot = args.plot

    # this can be controlled by command line parameter 'mode' (see above)
    # set drone.planner to Planner.VORONOI to use graph search mode
    # set drone.planner to Planner.GRID_SEARCH to use grid search mode (default)
    if args.mode == "voronoi":
        drone.planner = Planner.VORONOI
        print("mode = VORONOI")
    else:
        drone.planner = Planner.GRID_SEARCH
        print("mode = GRID_SEARCH")

    # set loiter_seconds to > 0 to loiter at waypoints
    drone.loiter_seconds = 0

    time.sleep(1)

    # SET GOAL
    # near davis and washington : 37.7961938,-122.3987356
    drone.set_goal(37.7961000, -122.3987356)

    drone.start()
