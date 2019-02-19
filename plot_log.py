from udacidrone import Drone
import matplotlib.pyplot as plt
from udacidrone.connection import MavlinkConnection

conn = MavlinkConnection('tcp:{0}:{1}'.format("127.0.0.1", 5760), timeout=1200)
drone = Drone(conn)

log = drone.read_telemetry_data("Logs/TLog.txt")
print(log)
