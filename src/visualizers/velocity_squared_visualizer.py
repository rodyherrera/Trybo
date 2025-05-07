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
        plt.show()

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