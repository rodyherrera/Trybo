from core.base_parser import BaseParser
from analyzers.centro_symmetric_analyzer import CentroSymmetricAnalyzer
from utilities.analyzer import get_atom_group_indices
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class CentroSymmetricVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = CentroSymmetricAnalyzer(parser)
        # Color map for centro-symmetric parameter
        self.centro_symmetric_cmap = 'viridis'
        # Colors for different structure classifications
        self.structure_colors = {
            'perfect': 'blue',
            'partial_defect': 'cyan',
            'stacking_fault': 'green',
            'surface': 'yellow',
            'defect': 'red'
        }
    
    def plot_centro_symmetric_distribution(self, timestep_idx=-1, group=None, log_scale=False):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]

        if group is not None and group != 'all':
            group_indices = analyzer.get_atom_group_indices(self.parser, timestep_idx)[group]
            data = data[group_indices]

        centro_symmetric_values = data[:, 5]
        plt.figure(figsize=(12, 8))
        ax = sns.histplot(centro_symmetric_values, kde=True, bins=50)
        # Vertical lines for classification thresholds
        for struct_type, (min_value, max_value) in self.analyzer.structure_ranges.items():
            if min_value > 0:
                plt.axvline(min_value, color=self.structure_colors.get(struct_type, 'gray'), linestyle='--', alpha=0.7, label=f'{struct_type} threshold: {min_value}')
        if log_scale:
            ax.set_yscale('log')
        # Add annotations for structure classifications
        y_max = ax.get_ylim()[1]
        y_pos = y_max * 0.9
        for struct_type, (min_value, max_value) in self.analyzer.structure_ranges.items():
            middle_value = (min_value + max_value) / 2
            if max_value < float('inf'):
                plt.annotate(struct_type, xy=(middle_value, y_pos), xytext=(middle_value, y_pos), ha='center', color=self.structure_colors.get(struct_type, 'gray'))
        
        stats = self.analyzer.get_defect_statistics(timestep_idx, group)
        stats_text = (f"Mean: {stats['mean']:.3f}\n"
                f"Max: {stats['max']:.3f}\n"
                f"Perfect crystal: {stats['perfect_percent']:.2f}%\n"
                f"Defect: {stats['defect_percent']:.2f}%\n"
                f"Stacking fault: {stats['stacking_fault_percent']:.2f}%")
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
               verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xlabel('Centro-Symmetric Parameter')
        plt.ylabel('Frequency')
        title = 'Centro-Symmetric Parameter Distribution'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        plt.title(f'{title} (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'cs_distribution_timestep_{current_timestep}.png', dpi=300)
    
    def plot_defect_evolution(self, group=None):
        timesteps, evolution = self.analyzer.get_defect_evolution(group)
        
        # Plot defect percentage
        plt.figure(figsize=(12, 8))
        plt.plot(timesteps, evolution['defect_percent'], 'r-', label='Defects')
        plt.plot(timesteps, evolution['perfect_percent'], 'b-', label='Perfect Crystal')
        plt.plot(timesteps, evolution['stacking_fault_percent'], 'g-', label='Stacking Faults')
         
        plt.xlabel('Timestep')
        plt.ylabel('Percentage of Atoms (%)')
        title = 'Evolution of Crystal Structure'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('defect_evolution.png', dpi=300)

        # Second plot for CS parameter values
        plt.figure(figsize=(12, 8))
        
        plt.plot(timesteps, evolution['mean'], 'b-', label='Mean Centro-Symmetric Value')
        plt.plot(timesteps, evolution['max'], 'r-', label='Max Centro-Symmetric Value')
        
        plt.xlabel('Timestep')
        plt.ylabel('Centro-Symmetric Parameter')
        title = 'Evolution of Centro-Symmetric Parameter'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        plt.savefig('cs_values_evolution.png', dpi=300)

    def plot_defect_3d(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        if group is not None and group != 'all':
            group_indices = get_atom_group_indices(self.parser, timestep_idx)[group]
            data = data[group_indices]
        
        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        cs_values = data[:, 5]
        
        classifications = self.analyzer.classify_atoms(cs_values)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for struct_type, mask in classifications.items():
            if np.any(mask):
                ax.scatter(x[mask], y[mask], z[mask], c=self.structure_colors.get(struct_type, 'gray'), s=10, alpha=0.7, label=struct_type)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')

        title = '3D Crystal Structure Visualization'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        ax.set_title(f'{title} (Timestep {current_timestep})')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'defect_3d_timestep_{current_timestep}.png', dpi=300)

    def plot_defect_regions(self, timestep_idx=-1, threshold=None, group=None):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        
        # Get full data for this timestep
        data = self.parser.get_data()[timestep_idx]

        # Get defect regions
        if group is not None and group != 'all':
            group_indices = get_atom_group_indices(self.parser, timestep_idx)[group]
            filtered_data = data[group_indices]
            defect_data, defect_mask = self.analyzer.get_defect_regions(timestep_idx, threshold, group)
            all_x, all_y, all_z = self.parser.get_atoms_spatial_coordinates(filtered_data)
        else:
            defect_data, defect_mask = self.analyzer.get_defect_regions(timestep_idx, threshold)
            all_x, all_y, all_z = self.parser.get_atoms_spatial_coordinates(data)

        defect_x, defect_y, defect_z = self.parser.get_atoms_spatial_coordinates(defect_data)
        defect_centro_symmetric = defect_data[:, 5]
                
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(all_x, all_y, all_z, c='lightgray', s=5, alpha=0.1)
        scatter = ax.scatter(defect_x, defect_y, defect_z, c=defect_centro_symmetric, cmap=self.centro_symmetric_cmap, s=30, alpha=1.0)
        plt.colorbar(scatter, ax=ax, label='Centro-Symmetric Parameter')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        title = 'Crystal Defect Regions'
        if threshold is not None:
            title += f' (CS ≥ {threshold})'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        ax.set_title(f'{title} (Timestep {current_timestep})')
        plt.tight_layout()
        plt.savefig(f'defect_regions_timestep_{current_timestep}.png', dpi=300)

    def plot_centro_symmetric_heatmaps(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        
        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        cs_values = data[:, 5]
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        bins = 50
        hxy = axs[0].hexbin(x, y, C=cs_values, gridsize=bins, reduce_C_function=np.mean, cmap=self.centro_symmetric_cmap)
        axs[0].set_title('Centro-Symmetric Parameter - XY Plane (Top View)')
        axs[0].set_xlabel('X (Å)')
        axs[0].set_ylabel('Y (Å)')
        fig.colorbar(hxy, ax=axs[0], label='Average CS Value')
        
        hxz = axs[1].hexbin(x, z, C=cs_values, gridsize=bins, reduce_C_function=np.mean, cmap=self.centro_symmetric_cmap)
        axs[1].set_title('Centro-Symmetric Parameter - XZ Plane (Side View)')
        axs[1].set_xlabel('X (Å)')
        axs[1].set_ylabel('Z (Å)')
        fig.colorbar(hxz, ax=axs[1], label='Average CS Value')
        
        hyz = axs[2].hexbin(y, z, C=cs_values, gridsize=bins, reduce_C_function=np.mean, cmap=self.centro_symmetric_cmap)
        axs[2].set_title('Centro-Symmetric Parameter - YZ Plane (Front View)')
        axs[2].set_xlabel('Y (Å)')
        axs[2].set_ylabel('Z (Å)')
        fig.colorbar(hyz, ax=axs[2], label='Average CS Value')
        
        plt.suptitle(f'Centro-Symmetric Parameter Heat Maps (Timestep {current_timestep})', y=1.05)
        plt.tight_layout()
        plt.savefig(f'cs_heatmaps_timestep_{current_timestep}.png', dpi=300)

    def plot_defect_by_groups(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        
        # Get statistics for each group
        nano_stats = self.analyzer.get_defect_statistics(timestep_idx, 'nanoparticle')
        upper_stats = self.analyzer.get_defect_statistics(timestep_idx, 'upper_plane')
        lower_stats = self.analyzer.get_defect_statistics(timestep_idx, 'lower_plane')
        
        # Prepare data for bar chart
        groups = ['Nanoparticle', 'Upper Plane', 'Lower Plane']
        perfect = [nano_stats['perfect_percent'], upper_stats['perfect_percent'], lower_stats['perfect_percent']]
        defect = [nano_stats['defect_percent'], upper_stats['defect_percent'], lower_stats['defect_percent']]
        stacking_fault = [nano_stats['stacking_fault_percent'], upper_stats['stacking_fault_percent'], lower_stats['stacking_fault_percent']]
        
        x = np.arange(len(groups))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        bar1 = ax.bar(x - width, perfect, width, label='Perfect Crystal', color=self.structure_colors['perfect'])
        bar2 = ax.bar(x, stacking_fault, width, label='Stacking Faults', color=self.structure_colors['stacking_fault'])
        bar3 = ax.bar(x + width, defect, width, label='Defects', color=self.structure_colors['defect'])
        
        ax.set_xlabel('Group')
        ax.set_ylabel('Percentage of Atoms (%)')
        ax.set_title(f'Comparison of Crystal Structure Between Groups (Timestep {current_timestep})')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend()
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords='offset points',
                          ha='center', va='bottom')
        
        add_labels(bar1)
        add_labels(bar2)
        add_labels(bar3)
        
        plt.tight_layout()
        plt.savefig(f'defect_by_groups_timestep_{current_timestep}.png', dpi=300)

    def plot_defect_profile(self, timestep_idx=-1, axis='z'):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        current_timestep = timesteps[timestep_idx]
        bin_centers, defect_percent, average_centro_symmetric = self.analyzer.calculate_defect_profile(timestep_idx, axis)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(bin_centers, defect_percent, 'r-o')
        ax1.set_ylabel('Defect Percentage (%)')
        ax1.set_title(f'Defect Profile Along {axis.upper()} Axis (Timestep {current_timestep})')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(bin_centers, average_centro_symmetric, 'b-o')
        ax2.set_xlabel(f'Position on {axis.upper()} Axis (Å)')
        ax2.set_ylabel('Average Centro Symmetric Value')
        ax2.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'defect_profile_{axis}_timestep_{current_timestep}.png', dpi=300)