from core.base_parser import BaseParser
from analyzers.velocity_squared_analyzer import VelocitySquaredAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class VelocitySquaredVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = VelocitySquaredAnalyzer(parser)
        self.temp_cmap = 'plasma'
    
    def plot_temperature_distribution(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            data = data[group_indices]
        velocity_squared = data[:, 5]
        temperature = self.analyzer.velocity_to_temperature(velocity_squared)
        plt.figure(figsize=(10, 8))
        sns.histplot(temperature, kde=True, bins=50)
        plt.axvline(np.mean(temperature), color='red', linestyle='--', label=f'Mean: {np.mean(temperature):.2f} K')
        plt.axvline(np.median(temperature), color='blue', linestyle='-.', label=f'Median: {np.median(temperature):.2f} K')
        stats = (f'Mean: {np.mean(temperature):.2f} K\n'
                f'Median: {np.median(temperature):.2f} K\n'
                f'Max: {np.max(temperature):.2f} K\n'
                f'Min: {np.min(temperature):.2f} K\n'
                f'STD: {np.std(temperature):.2f} K')
        plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, 
               verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.xlabel('Temperature (K)')
        plt.ylabel('Frequency')
        title = f'Temperature Distribution (Timestep {current_timestep})'
        if group is not None and group != 'all':
            title += f' - Group {group}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'temperature_distribution_timestep_{current_timestep}.png', dpi=300)

    def plot_temperature_evolution(self, group=None):
        timesteps, average_temperature, max_temperature, min_temperature= self.analyzer.get_temperature_evolution(group)
        plt.figure(figsize=(12, 8))
        
        plt.plot(timesteps, average_temperature, 'b-', label='Average Temperature')
        plt.plot(timesteps, max_temperature, 'r-', label='Maximum Temperature')
        plt.fill_between(timesteps, min_temperature, max_temperature, color='blue', alpha=0.2, label='Min-Max Range')
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        title = 'Temperature Evolution'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('temperature_evolution.png', dpi=300)
    
    def plot_temperature_3d(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            data = data[group_indices]
        x = data[:, 2]
        y = data[:, 3]
        z = data[:, 4]
        velocity_squared = data[:, 5]
        temperature = self.analyzer.velocity_to_temperature(velocity_squared)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=temperature, cmap=self.temp_cmap, s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Temperature (K)')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        title = f'3D Temperature Distribution (Timestep {current_timestep})'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f'temperature_3d_timestep_{current_timestep}.png', dpi=300)
    
    def plot_hot_spots(self, timestep_idx=-1, threshold_percentile=95, group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        current_timestep = timesteps[timestep_idx]
        data = self.parser.get_data()[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            filtered_data = data[group_indices]
            hot_spots_data, hot_spots_mask = self.analyzer.get_hot_spots(timestep_idx, threshold_percentile, group)
            all_x = filtered_data[:, 2]
            all_y = filtered_data[:, 3]
            all_z = filtered_data[:, 4]
        else:
            hot_spots_data, hot_spots_mask = self.analyzer.get_hot_spots(timestep_idx, threshold_percentile)
            all_x = data[:, 2]
            all_y = data[:, 3]
            all_z = data[:, 4]
        hot_x = hot_spots_data[:, 2]
        hot_y = hot_spots_data[:, 3]
        hot_z = hot_spots_data[:, 4]
        hot_velocity_squared = hot_spots_data[:, 5]
        hot_temperature = self.analyzer.velocity_to_temperature(hot_velocity_squared)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(all_x, all_y, all_z, c='lightgray', s=5, alpha=0.1)
        scatter = ax.scatter(hot_x, hot_y, hot_z, c=hot_temperature, cmap=self.temp_cmap, s=30, alpha=1.0)
        plt.colorbar(scatter, ax=ax, label='Temperature (K)')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        title = f'Puntos Calientes (>{threshold_percentile}%) - Timestep {current_timestep}'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f'hot_spots_timestep_{current_timestep}.png', dpi=300)
    
    def plot_temperature_heatmaps(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        x = data[:, 2]
        y = data[:, 3]
        z = data[:, 4]
        velocity_squared = data[:, 5]
        temperature = self.analyzer.velocity_to_temperature(velocity_squared)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        bins = 50
        hxy = axs[0].hexbin(x, y, C=temperature, gridsize=bins, reduce_C_function=np.mean, cmap=self.temp_cmap)
        axs[0].set_title('Temperature Map - XY Plane (Top View)')
        axs[0].set_xlabel('X (Å)')
        axs[0].set_ylabel('Y (Å)')
        fig.colorbar(hxy, ax=axs[0], label='Average Temperature (K)')

        hxz = axs[1].hexbin(x, z, C=temperature, gridsize=bins, reduce_C_function=np.mean, cmap=self.temp_cmap)
        axs[1].set_title('Temperature Map - XZ Plane (Side View)')
        axs[1].set_xlabel('X (Å)')
        axs[1].set_ylabel('Z (Å)')
        fig.colorbar(hxz, ax=axs[1], label='Average Temperature (K)')

        hyz = axs[2].hexbin(y, z, C=temperature, gridsize=bins, reduce_C_function=np.mean, cmap=self.temp_cmap)
        axs[2].set_title('Temperature Map - YZ Plane (Front View)')
        axs[2].set_xlabel('Y (Å)')
        axs[2].set_ylabel('Z (Å)')
        fig.colorbar(hyz, ax=axs[2], label='Average Temperature (K)')

        plt.suptitle(f'Temperature Heat Steps - Timestep {current_timestep}', y=1.05)
        plt.savefig(f'temperature_heatmaps_timestep_{current_timestep}.png', dpi=300)
        
    def plot_temperature_by_groups(self):
        timesteps, nano_average, nano_max, nano_min = self.analyzer.get_temperature_evolution('nanoparticle')
        _, upper_average, _, _ = self.analyzer.get_temperature_evolution('upper_plane')
        _, lower_average, _, _ = self.analyzer.get_temperature_evolution('lower_plane')
        plt.figure(figsize=(12, 8))
                
        plt.plot(timesteps, nano_average, 'r-', label='Nanoparticle')
        plt.plot(timesteps, upper_average, 'g-', label='Lower Plane')
        plt.plot(timesteps, lower_average, 'b-', label='Upper Plane')
        
        plt.fill_between(timesteps, nano_min, nano_max, color='red', alpha=0.2)

        plt.xlabel('Timestep')
        plt.ylabel('Average Temperature (K)')
        plt.title('Comparison of Temperature between Groups')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('temperature_by_groups.png', dpi=300)
    
    def plot_temperature_gradient(self, timestep_idx=-1, axis='z'):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        current_timestep = timesteps[timestep_idx]
        bin_centers, bin_temps = self.analyzer.calculate_temperature_gradient(timestep_idx, axis)
        plt.figure(figsize=(12, 8))
        plt.plot(bin_centers, bin_temps, 'r-o')
        plt.xlabel(f'Position on axis {axis.upper()} (Å)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Temperature gradient along axis {axis.upper()} (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'temperature_gradient_{axis}_timestep_{current_timestep}.png', dpi=300)