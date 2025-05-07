from core.base_parser import BaseParser
import numpy as np

class VelocitySquaredAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None
        # Conversion factor for metal units
        self.metal_units_conversion = 25.464

    def get_atom_group_indices(self):
        if self._atom_groups is not None:
            return self._atom_groups
        data = self.parser.get_data()[0]
        # Groups
        # TODO: Duplicated code (ptm analyzer)
        # x = data[:, 2]
        # y = data[:, 3]
        z = data[:, 4]
        # Dimensions
        z_min = np.min(z)
        z_max = np.max(z)
        z_threshold_lower = z_min + 2.5
        z_threshold_upper = z_max - 2.5
        # Indexes for each group
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

    def velocity_to_temperature(self, velocity_squared):
        # T = m * v² / (3 * k_B)
        return velocity_squared * self.metal_units_conversion
    
    def get_hot_spots(self, timestep_idx=-1, threshold_percentile=95, group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.get_atom_group_indices()[group]
            data = data[group_indices]
        velocity_squared = data[:, 5]
        hot_threshold = np.percentile(velocity_squared, threshold_percentile)
        hot_spots_mask = velocity_squared >= hot_threshold
        hot_spots_data = data[hot_spots_mask]
        return hot_spots_data, hot_spots_mask
    
    def get_temperature_evolution(self, group=None):
        timesteps = self.parser.get_timesteps()
        all_data = self.parser.get_data()
        average_temperature = []
        max_temperature = []
        min_temperature = []
        for idx, data in enumerate(all_data):
            if group is not None and group != 'all':
                group_indices = self.get_atom_group_indices()[group]
                current_data = data[group_indices]
            else:
                current_data = data
            velocity_squared = current_data[:, 5]
            temperature = self.velocity_to_temperature(velocity_squared)
            average_temperature.append(np.mean(temperature))
            max_temperature.append(np.max(temperature))
            min_temperature.append(np.min(temperature))
        return timesteps, average_temperature, max_temperature, min_temperature

    def get_temperature_statistics(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.get_atom_group_indices()[group]
            data = data[group_indices]
        velocity_squared = data[:, 5]
        temperature = self.velocity_to_temperature(velocity_squared)
        stats = {
            'mean': np.mean(temperature),
            'median': np.median(temperature),
            'max': np.max(temperature),
            'min': np.min(temperature),
            'std': np.std(temperature)
        }
        
        return stats