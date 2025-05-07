def get_data_from_coord_axis(axis, coords):
    x, y, z = coords
    values = {
        'x': x,
        'y': y,
        'z': z
    }
    return values.get(axis, z)