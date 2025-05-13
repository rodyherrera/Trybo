from core.base_parser import BaseParser
from utilities.analyzer import get_data_from_coord_axis, get_atom_group_indices
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
    
    def classify_atoms(self, centro_symmetric_values):
        classifications = {}
        for struct_type, (min_value, max_value) in self.structure_ranges.items():
            mask = (centro_symmetric_values >= min_value) & (centro_symmetric_values < max_value)
            classifications[struct_type] = mask
        return classifications

    def get_defect_statistics(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timesteps
        
        centro_symmetric_values = self.parser.get_analysis_data('centro_symmetric', timestep_idx)

        if group is not None and group != 'all':
            data = self.parser.get_data(timestep_idx)
            group_indices = self.parser.get_atom_group_indices(data)[group]
            centro_symmetric_values = centro_symmetric_values[group_indices]

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
    
    def get_defect_evolution(self, group=None):
        timesteps = self.parser.get_timesteps()
        all_data = self.parser.get_data()
        evolution = {
            'mean': [],
            'max': [],
            'defect_percent': [],
            'perfect_percent': [],
            'stacking_fault_percent': []
        }
        for idx in range(len(timesteps)):
            centro_symmetric_values = self.parser.get_analysis_data('centro_symmetric', idx)

            if group is not None and group != 'all':
                data = self.parser.get_data(idx)
                group_indices = self.parser.get_atom_group_indices(data)[group]
                centro_symmetric_values = centro_symmetric_values[group_indices]
        
            classifications = self.classify_atoms(centro_symmetric_values)
            evolution['mean'].append(np.mean(centro_symmetric_values))
            evolution['max'].append(np.max(centro_symmetric_values))
            total_atoms = len(centro_symmetric_values)
            evolution['defect_percent'].append(np.sum(classifications['defect']) * 100 / total_atoms)
            evolution['perfect_percent'].append(np.sum(classifications['perfect']) * 100 / total_atoms)
            evolution['stacking_fault_percent'].append(np.sum(classifications['stacking_fault']) * 100 / total_atoms)
        return timesteps, evolution

    def get_defect_regions(self, timestep_idx=-1, threshold=None, group=None):
        if threshold is None:
            threshold = self.defect_threshold
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        centro_symmetric_values = self.parser.get_analysis_data('centro_symmetric', timestep_idx)
        data = self.parser.get_data(timestep_idx)
        if group is not None and group != 'all':
            group_indices = self.parser.get_atom_group_indices(data)[group]
            centro_symmetric_values = centro_symmetric_values[group_indices]
            data = data[group_indices]
        defect_mask = centro_symmetric_values >= threshold
        defect_data = data[defect_mask]
        return defect_data, defect_mask

    def calculate_defect_profile(self, timestep_idx=-1, axis='z', n_bins=20):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data(timestep_idx)
        atoms_spatial_coordinates = self.parser.get_atoms_spatial_coordinates(data)
        coords = get_data_from_coord_axis(axis, atoms_spatial_coordinates)
        centro_symmetric_values = self.parser.get_analysis_data('centro_symmetric', timestep_idx)
        bins = np.linspace(np.min(coords), np.max(coords), n_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        defect_percent = []
        average_centro_symmetric = []
        for i in range(n_bins):
            bin_min = bins[i]
            bin_max = bins[i + 1]
            # Atoms in this bin
            bin_mask = (coords >= bin_min) & (coords < bin_max)
            bin_centro_symmetric_values = centro_symmetric_values[bin_mask]
            if len(bin_centro_symmetric_values) > 0:
                defect_count = np.sum(bin_centro_symmetric_values >= self.defect_threshold)
                defect_percent.append(defect_count * 100 / len(bin_centro_symmetric_values))
                average_centro_symmetric.append(np.mean(bin_centro_symmetric_values))
            else:
                defect_percent.append(0)
                average_centro_symmetric.append(0)
        return bin_centers, defect_percent, average_centro_symmetric