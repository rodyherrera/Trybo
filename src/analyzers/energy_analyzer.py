from core.base_parser import BaseParser
import numpy as np

class EnergyAnaylizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None

    def get_atom_group_indices(self):
        if self._atom_groups is not None:
            return self._atom_groups

        data = self.parser.get_data()[0]
        # x = data[:, 2]
        # y = data[:, 3]
        z = data[:, 4]

        z_min = np.min(z)
        z_max = np.max(z)
        z_threshold_lower = z_min + 2.5 
        z_threshold_upper = z_max - 2.5 
        
        # Indices for each group
        lower_plane_mask = z <= z_threshold_lower
        upper_plane_mask = z >= z_threshold_upper
        nanoparticle_mask = ~(lower_plane_mask | upper_plane_mask)
        
        self._atom_groups = {
            'lower_plane': np.where(lower_plane_mask)[0],
            'upper_plane': np.where(upper_plane_mask)[0],
            'nanoparticle': np.where(nanoparticle_mask)[0],
            'all': np.arange(len(data))
        }
        
        return self._atom_groups