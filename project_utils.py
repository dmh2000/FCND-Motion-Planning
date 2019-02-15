import numpy as np
import re

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


def prune_path(grid, path):
    return path


# test
if __name__ == "__main__":
    lat, lon, data = read_environment()
    print(lat, lon, len(data))
