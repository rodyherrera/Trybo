def get_data_from_coord_axis(axis, coords):
    x, y, z = coords
    values = {
        'x': x,
        'y': y,
        'z': z
    }
    return values.get(axis, z)

def get_atom_group_indices(parser, timestep_idx):
    data = parser.get_data()[timestep_idx]
    atom_groups = parser.get_atom_group_indices(data)
    return atom_groups