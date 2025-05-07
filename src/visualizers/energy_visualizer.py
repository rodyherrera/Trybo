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

    def plot_energy_evolution(self, group=None, energy_type='total'):
        timesteps, average_energy, max_energy, min_energy, sum_energy = self.analyzer.get_energy_evolution(group, energy_type)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        if energy_type == 'kinetic':
            title_prefix = 'Kinetic'
            y_label = 'Kinetic Energy (eV)'
        elif energy_type == 'potential':
            title_prefix = 'Potential'
            y_label = 'Potential Energy (eV)'
        else: 
            title_prefix = 'Total'
            y_label = 'Total Energy (eV)'
        ax1.plot(timesteps, average_energy, 'b-', label='Average Energy')
        ax1.plot(timesteps, max_energy, 'r-', label='Maximum Energy')
        ax1.plot(timesteps, min_energy, 'g-', label='Minimum Energy')
        ax1.fill_between(timesteps, min_energy, max_energy, color='blue', alpha=0.2, label='Min-Max Range')
        
        ax1.set_ylabel(f'Average {y_label}')
        ax1.set_title(f'Evolution of Average {title_prefix} Energy')
        if group is not None and group != 'all':
            ax1.set_title(f'Evolution of Average {title_prefix} Energy - Group: {group}')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        ax2.plot(timesteps, sum_energy, 'g-', label='System Energy')
        
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel(f'Total System {y_label}')
        ax2.set_title(f'Evolution of System {title_prefix} Energy')
        if group is not None and group != 'all':
            ax2.set_title(f'Evolution of System {title_prefix} Energy - Group: {group}')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        plt.savefig(f'{energy_type}_energy_evolution.png', dpi=300)
        
        plt.show()
    
    def plot_energy_3d(self, timestep_idx=-1, group=None, energy_type='total'):
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
        
        if energy_type == 'kinetic':
            energy_col = 5
            title_prefix = 'Kinetic'
            cmap = self.energy_cmaps['kinetic']
        elif energy_type == 'potential':
            energy_col = 6 
            title_prefix = 'Potential'
            cmap = self.energy_cmaps['potential']
        else:
            energy_col = 7
            title_prefix = 'Total'
            cmap = self.energy_cmaps['total']
        
        energy_values = data[:, energy_col]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x, y, z, c=energy_values, cmap=cmap, s=10, alpha=0.7)
        
        cbar = plt.colorbar(scatter, ax=ax, label=f'{title_prefix} Energy (eV)')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        
        title = f'3D {title_prefix} Energy Distribution (Timestep {current_timestep})'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        ax.set_title(title)
        
        plt.tight_layout()
        
        plt.savefig(f'{energy_type}_energy_3d_timestep_{current_timestep}.png', dpi=300)