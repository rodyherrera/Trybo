from core.base_parser import BaseParser
from analyzers.energy_visualizer import EnergyAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EnergyVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = EnergyAnalyzer(parser)

        # Color maps for different energy types
        self.energy_cmaps = {
            'kinetic': 'plasma',
            'potential': 'viridis', 
            'total': 'turbo'
        }
    
    def plot_energy_distribution(self, timestep_idx=-1, group=None, energy_type='total'):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            data = data[group_indices]
        if energy_type == 'kinetic':
            # c_ke_mobile
            energy_col = 5
            title_prefix = 'Kinetic'
            x_label = 'Kinetic Energy (eV)'
        elif energy_type == 'potential':
            # c_pe_mobile
            energy_col = 6
            title_prefix = 'Potential'
            x_label = 'Potential Energy (eV)'
        else:
            # v_total_energy
            energy_col = 7
            title_prefix = 'Total'
            x_label = 'Total Energy (eV)'
        energy_values = data[:, energy_values]
        plt.figure(figsize=(10, 8))
        sns.histplot(energy_values, kde=True, bins=50)
        plt.axvline(np.mean(energy_values), color='red', linestyle='--', label=f'Mean: {np.mean(energy_values):.3f} eV')
        plt.axvline(np.median(energy_values), color='blue', linestyle='-.', label=f'Median: {np.median(energy_values):.3f} eV')
        
        stats = (f'Mean: {np.mean(energy_values):.3f} eV\n'
            f'Median: {np.median(energy_values):.3f} eV\n'
            f'Max: {np.max(energy_values):.3f} eV\n'
            f'Min: {np.min(energy_values):.3f} eV\n'
            f'Std Dev: {np.std(energy_values):.3f} eV\n'
            f'Total: {np.sum(energy_values):.3f} eV')

        plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, 
               verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.xlabel(x_label)
        plt.ylabel('Frequency')
        title = f'{title_prefix} Energy Distribution (Timestep {current_timestep})'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{energy_type}_energy_distribution_timestep_{current_timestep}.png', dpi=300)