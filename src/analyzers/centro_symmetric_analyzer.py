from core.base_parser import BaseParser
import numpy as np

class CentroSymmetricAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None
        # Defect thresholds for FCC (face-centered cubic) copper
        # Below this is considered perfect crystal
        self.perfect_threshold = 0.5
        # Above this is considered a significant defect
        self.defect_threshold = 8.0
        # Typical range for stacking faults in FCC
        self.stacking_fault_range = (2.0, 5.0)
        # Classification thresholds
        self.structure_ranges = {
            'perfect': (0, 0.5),
            'partial_defect': (0.5, 3.0),
            'stacking_fault': (3.0, 5.0),
            'surface': (5.0, 8.0),
            'defect': (8.0, float('inf'))
        }
    
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
