import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


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


# NED COORDINATE SYSTEM
# right handed (down is positive)
# X is positive north (up in simulator)
# Y is positive east  (right in simulator)
# Z is psitive down

class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.in_mission = True
        self.check_state = {}

        # set waypoint parameters
        self.waypoints = []  # list of waypoints in flight order

        # initial state
        self.flight_state = States.MANUAL

        # register callbacks
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    # ===============================================================
    # callbacks dispatch event to state handler
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
        self.state_handler(Events.LOCAL_POSITION)

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
        """T
        """
        print("arming transition")
        self.arm()
        self.take_control()
        self.flight_state = States.ARMING

    def takeoff_transition(self):
        """
        """
        print("takeoff transition")
        self.takeoff(self.target_position[2])
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        """
        1. Command the next waypoint position
        2. Transition to WAYPOINT state
        """
        print("waypoint transition")
        # command next waypoint : correct Z to positive up (waypoint is ned)
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0],
                          self.target_position[1],
                          self.target_position[2],
                          self.target_position[3])

        self.flight_state = States.WAYPOINT

    def landing_transition(self):
        """T
        1. Command the drone to land
        2. Transition to the LANDING state
        """
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        """
        1. Command the drone to disarm
        2. Transition to the DISARMING state
        """
        print("disarm transition")
        self.disarm()
        self.release_control()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        """
        1. Release control of the drone
        2. Stop the connection (and telemetry log)
        3. End the mission
        4. Transition to the MANUAL state
        """
        print("manual transition")
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
            # wait until armed then initiate takeoff
            if self.armed:
                self.plan_path()
        else:
            print("INVALID EVENT", self.flight_state, event)

    def planning_state(self, event):
        """
        :param event:
        :return:
        """
        if event == Events.LOCAL_POSITION:
            pass
        elif event == Events.VELOCITY:
            pass
        elif event == Events.STATE:
            # wait until armed then initiate takeoff
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
            altitude = -1.0 * self.local_position[2]

            # check if altitude is within 95% of target
            if altitude > 0.95 * self.target_position[2]:
                # go to landing
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
        if event == Events.LOCAL_POSITION:
            # check waypoint bounds and go to next
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
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
        dispatch events to current state
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
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values

        # TODO: set home position to (lon0, lat0, 0)

        # TODO: retrieve current global position

        # TODO: convert to current local position using global_to_local()

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center

        # Set goal as some arbitrary position on the grid
        grid_goal = (-north_offset + 10, -east_offset + 10)
        # TODO: adapt to set goal as latitude / longitude position and convert

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()

    def start(self):
        """This method is provided
        
        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--port', type=int, default=5760, help='Port number')
        parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
        args = parser.parse_args()

        conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
        drone = MotionPlanning(conn)
        time.sleep(1)

        drone.start()
