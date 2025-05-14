from core.base_parser import BaseParser
from utilities.analyzer import get_data_from_coord_axis
import numpy as np
import cupy as cp

class VelocitySquaredAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None
        # Conversion factor for metal units
        self.metal_units_conversion = 25.464

    def velocity_to_temperature(self, velocity_squared):
        # T = m * v² / (3 * k_B)
        return velocity_squared * self.metal_units_conversion
    
    def get_hot_spots(self, timestep_idx=-1, threshold_percentile=95, group=None):
        velocity_squared = self.parser.get_analysis_data('velocity_squared', timestep_idx)
        data = self.parser.get_data(timestep_idx)
        if group is not None and group != 'all':
            group_indices = self.parser.get_atom_group_indices(data)[group]
            velocity_squared = velocity_squared[group_indices]
            data = data[group_indices]
        hot_threshold = np.percentile(velocity_squared, threshold_percentile)
        hot_spots_mask = velocity_squared >= hot_threshold
        hot_spots_data = data[hot_spots_mask]
        return hot_spots_data, hot_spots_mask
    
    def get_temperature_evolution(self, group=None):
        timesteps = self.parser.get_timesteps()
        average_temperature = []
        max_temperature = []
        min_temperature = []
        for idx in range(len(timesteps)):
            velocity_squared = self.parser.get_analysis_data('velocity_squared', idx)
            if group is not None and group != 'all':
                data = self.parser.get_data(idx)
                group_indices = self.parser.get_atom_group_indices(data)[group]
                velocity_squared = velocity_squared[group_indices]
            temperature = self.velocity_to_temperature(velocity_squared)
            average_temperature.append(np.mean(temperature))
            max_temperature.append(np.max(temperature))
            min_temperature.append(np.min(temperature))
        return timesteps, average_temperature, max_temperature, min_temperature

    def get_temperature_statistics(self, timestep_idx=-1, group=None):
        velocity_squared = self.parser.get_analysis_data('velocity_squared', timestep_idx)
        data = self.parser.get_data(timestep_idx)
        if group is not None and group != 'all':
            group_indices = self.parser.get_atom_group_indices(data)[group]
            velocity_squared = velocity_squared[group_indices]
        temperature = self.velocity_to_temperature(velocity_squared)
        stats = {
            'mean': np.mean(temperature),
            'median': np.median(temperature),
            'max': np.max(temperature),
            'min': np.min(temperature),
            'std': np.std(temperature)
        }
        
        return stats

    def calculate_temperature_gradient(self, timestep_idx=-1, axis='z', n_bins=20):
        timesteps = self.parser.get_timesteps()
        data = self.parser.get_data(timestep_idx)
        atoms_spatial_coordinates = self.parser.get_atoms_spatial_coordinates(data)
        coords = get_data_from_coord_axis(axis, atoms_spatial_coordinates)
        
        velocity_squared = self.parser.get_analysis_data('velocity_squared', timestep_idx)
        
        temperature = self.velocity_to_temperature(velocity_squared)
        bins = np.linspace(np.min(coords), np.max(coords), n_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        digitized = np.digitize(coords, bins)
        bin_temps = [temperature[digitized == i].mean() for i in range(1, len(bins))]
        
        return bin_centers, bin_temps