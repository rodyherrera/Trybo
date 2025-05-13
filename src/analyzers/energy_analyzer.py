from core.base_parser import BaseParser
from utilities.analyzer import get_data_from_coord_axis, get_atom_group_indices
import numpy as np

class EnergyAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None

    def get_energy_statistics(self, timestep_idx=-1, group=None, energy_type='total'):
        timesteps = self.parser.get_timesteps()
        energy_key = self.get_energy_column_by_type(energy_type)
        energy_values = self.parser.get_analysis_data(energy_key, timestep_idx)

        if group is not None and group != 'all':
            data = self.parser.get_data(timestep_idx)
            group_indices = self.parser.get_atom_group_indices(data)[group]
            energy_values = energy_values[group_indices]
                
        stats = {
            'mean': np.mean(energy_values),
            'median': np.median(energy_values),
            'max': np.max(energy_values),
            'min': np.min(energy_values),
            'std': np.std(energy_values),
            'sum': np.sum(energy_values)
        }
        
        return stats

    def get_energy_evolution(self, group=None, energy_type='total'):
        timesteps = self.parser.get_timesteps()
        energy_key = self.get_energy_column_by_type(energy_type)
        average_energy = []
        max_energy = []
        min_energy = []
        sum_energy = []
        for idx in range(len(timesteps)):
            energy_values = self.parser.get_analysis_data(energy_key, idx)
            
            if group is not None and group != 'all':
                data = self.parser.get_data(idx)
                group_indices = self.parser.get_atom_group_indices(data)[group]
                energy_values = energy_values[group_indices]

            average_energy.append(np.mean(energy_values))
            max_energy.append(np.max(energy_values))
            min_energy.append(np.min(energy_values))
            sum_energy.append(np.sum(energy_values))
        return timesteps, average_energy, max_energy, min_energy, sum_energy
    
    def get_high_energy_regions(self, timestep_idx=-1, threshold_percentile=95, energy_type='total', group=None):
        timesteps = self.parser.get_timesteps()
        data = self.parser.get_data(timestep_idx)
        if group is not None and group != 'all':
            group_indices = get_atom_group_indices(self.parser, timestep_idx)[group]
            data = data[group_indices]
        energy_col = self.get_energy_column_by_type(energy_type)
        energy_values = data[:, energy_col]
        # Calculate threshold for high energy
        # For potential energy, we're looking for the most negative values (most stable)
        if energy_type == 'potential':
            high_threshold = np.percentile(energy_values, 100 - threshold_percentile)
            high_energy_mask = energy_values <= high_threshold
        else:
            high_threshold = np.percentile(energy_values, threshold_percentile)
            high_energy_mask = energy_values >= high_threshold
        high_energy_data = data[high_energy_mask]
        return high_energy_data, high_energy_mask
    
    def get_energy_column_by_type(self, energy_type):
        types = {
            # c_ke_mobile
            'kinetic': 'ke_mobile',
            # c_pe_mobile
            'potential': 'pe_mobile',
            # v_total_energy
            'total': 'total_energy'
        }
        return types.get(energy_type, 'total_energy')

    def calculate_energy_profile(self, timestep_idx=-1, axis='z', n_bins=20, energy_type='total'):
        data = self.parser.get_data(timestep_idx)
        atoms_spatial_coordinates = self.parser.get_atoms_spatial_coordinates(data)
        coords = get_data_from_coord_axis(axis, atoms_spatial_coordinates)
        energy_col = self.get_energy_column_by_type(energy_type)
        energy_values = data[:, energy_col]
        # Create bins alongs the axis
        bins = np.linspace(np.min(coords), np.max(coords), n_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Calculate average energy for each bin
        digitized = np.digitize(coords, bins)
        bin_energies = [energy_values[digitized == i].mean() for i in range(1, len(bins))]
        
        return bin_centers, bin_energies