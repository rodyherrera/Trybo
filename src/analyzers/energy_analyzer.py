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
    
    def get_energy_statistics(self, timestep_idx=-1, group=None, energy_type='total'):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.get_atom_group_indices()[group]
            data = data[group_indices]
        if energy_type == 'kinetic':
            # c_ke_mobile
            energy_col = 5
        elif energy_type == 'potential':
            # c_pe_mobile
            energy_col = 6
        else:
            # v_total_energy
            energy_col = 7
        energy_values = data[:, energy_col]
                
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
        all_data = self.parser.get_data()
        # TODO: REFACTOR THIS DUPLICATED CODE
        if energy_type == 'kinetic':
            # c_ke_mobile
            energy_col = 5
        elif energy_type == 'potential':
            # c_pe_mobile
            energy_col = 6
        else:
            # v_total_energy
            energy_col = 7
        average_energy = []
        max_energy = []
        min_energy = []
        sum_energy = []
        for idx, data in enumerate(all_data):
            if group is not None and group != 'all':
                group_indices = self.get_atom_group_indices()[group]
                current_data = data[group_indices]
            else:
                current_data = data
            energy_values = current_data[:, energy_col]
            average_energy.append(np.mean(energy_values))
            max_energy.append(np.max(energy_values))
            min_energy.append(np.min(energy_values))
            sum_energy.append(np.sum(energy_values))
        return timesteps, average_energy, max_energy, min_energy, sum_energy
    
    def get_high_energy_regions(self, timestep_idx=-1, threshold_percentile=95, energy_type='total', group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.get_atom_group_indices()[group]
            data = data[group_indices]
        if energy_type == 'kinetic':
            # c_ke_mobile
            energy_col = 5
        elif energy_type == 'potential':
            # c_pe_mobile
            energy_col = 6
        else:
            # v_total_energy
            energy_col = 7
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
    