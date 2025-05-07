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

    def classify_atoms(self, centro_symmetric_values):
        classifications = {}
        for struct_type, (min_value, max_value) in self.structure_ranges.items():
            mask = (centro_symmetric_values >= min_value) & (centro_symmetric_values < max_value)
            classifications[struct_type] = mask
        return classifications

    def get_defect_statistics(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.get_atom_group_indices()[group]
            data = data[group_indices]
        
        centro_symmetric_values = data[:, 5]
        classifications = self.classify_atoms(centro_symmetric_values)
        total_atoms = len(centro_symmetric_values)
        stats = {
            'total_atoms': total_atoms,
            'mean': np.mean(centro_symmetric_values),
            'max': np.max(centro_symmetric_values),
            'min': np.min(centro_symmetric_values),
            'std': np.std(centro_symmetric_values),
            'percent_defect': np.sum(classifications['defect']) * 100 / total_atoms,
        }

        for struct_type, mask in classifications.items():
            count = np.sum(mask)
            stats[f'{struct_type}_count'] = count
            stats[f'{struct_type}_percent'] = count * 100 / total_atoms
        
        return stats