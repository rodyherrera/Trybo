import numpy as np

def get_coords(data):
    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]
    return x, y, z

def get_data_from_coord_axis(axis, data):
    x, y, z = get_coords(data)
    coords = {
        'x': x,
        'y': y,
        'z': z
    }
    return coords.get(axis, z), 