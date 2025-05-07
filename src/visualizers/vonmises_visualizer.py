from core.base_parser import BaseParser
from analyzers.vonmises_analyzer import VonMisesAnalyzer
from utilities.analyzer import get_data_from_coord_axis
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class VonmisesVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = VonMisesAnalyzer(parser)

    def plot_stress_evolution(self):
        timesteps = self.parser.get_timesteps()
        average_stress, max_stress, min_stress = self.analyzer.get_stress_evolution()
        
        plt.figure(figsize=(12, 8))
        plt.plot(timesteps, average_stress, 'b--', label='Average Stress')
        plt.plot(timesteps, max_stress, 'r--', label='Maximum Stress')
        plt.fill_between(timesteps, min_stress, max_stress, color='blue', alpha=0.2, label='Min-Max Range')
        plt.xlabel('Timestep')
        plt.ylabel('von Mises Stress')
        plt.title('Evolution of von Mises Stress')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('stress_evolution.png', dpi=300)

    def plot_stress_heatmaps(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx

        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]

        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        stress = data[:, 5]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        bins = 50

        hxy = axs[0].hexbin(x, y, C=stress, gridsize=bins, reduce_C_function=np.mean, cmap='hot')
        axs[0].set_title('Stress Map - XY Plane (Top View)')
        axs[0].set_xlabel('X (Å)')
        axs[0].set_ylabel('Y (Å)')
        fig.colorbar(hxy, ax=axs[0], label='Average Stress')

        hxz = axs[1].hexbin(x, z, C=stress, gridsize=bins, reduce_C_function=np.mean, cmap='hot')
        axs[1].set_title('Stress Map - XZ Plan (Side View)')
        axs[1].set_xlabel('X (Å)')
        axs[1].set_ylabel('Z (Å)')
        fig.colorbar(hxz, ax=axs[1], label='Average Stress')

        hyz = axs[2].hexbin(y, z, C=stress, gridsize=bins, reduce_C_function=np.mean, cmap='hot')
        axs[2].set_title('Stress Map - YZ Plane (Front View)')
        axs[2].set_xlabel('Y (Å)')
        axs[2].set_ylabel('Z (Å)')
        fig.colorbar(hyz, ax=axs[2], label='Average Stress')

        plt.suptitle(f'von Mises Stress Heat Maps - Timestep {current_timestep} - Timestep {current_timestep}', y=1.05)
        plt.tight_layout()
        plt.savefig(f'stress_heatmaps_timesteps_{current_timestep}.png', dpi=300)

    def plot_stress_distribution(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]
        stress = data[:, 5]
        current_timestep = timesteps[timestep_idx]

        plt.figure(figsize=(10, 8))
        sns.histplot(stress, kde=True, bins=50, color='orangered')
        plt.axvline(np.mean(stress), color='blue', linestyle='--', label=f'Mean: {np.mean(stress):.2e}')
        plt.axvline(np.median(stress), color='green', linestyle='--', label=f'Median: {np.median(stress):.2e}')
        plt.xlabel('von Mises Stress')
        plt.ylabel('Frequency')
        plt.title(f'Von Mises Stress Distribution (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        stats = (f'Mean: {np.mean(stress):.2e}\n'
                f'Median: {np.median(stress):.2e}\n'
                f'Max: {np.max(stress):.2e}\n'
                f'Min: {np.min(stress):.2e}\n'
                f'STD: {np.std(stress):.2e}')

        plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'stress_distribution_timestep_{current_timestep}.png', dpi=300)

    def plot_stress_by_groups(self):
        '''
        Compare the evolution of the average stress between the different groups (nanoparticle, upper and lower planes)
        '''
        timesteps = self.parser.get_timesteps()

        nanoparticle_average, nanoparticle_max, nanoparticle_min = self.analyzer.get_stress_evolution_by_group('nanoparticle')
        upper_average, _, _ = self.analyzer.get_stress_evolution_by_group('upper_plane')
        lower_average, _, _ = self.analyzer.get_stress_evolution_by_group('lower_plane')
        
        plt.figure(figsize=(12, 8))
        plt.plot(timesteps, nanoparticle_average, 'r--', label='Nanoparticle')
        plt.plot(timesteps, upper_average, 'g--', label='Upper plane')
        plt.plot(timesteps, lower_average, 'b--', label='Lower plane')
        plt.fill_between(timesteps, nanoparticle_min, nanoparticle_max, color='red', alpha=0.2)
        plt.xlabel('Timestep')
        plt.ylabel('Average von Mises Stress')
        plt.title('Comparison of Stress between Groups')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('stress_by_group.png', dpi=300)

    def plot_stress_3d(self, timestep_idx=-1, group=None, percentile_threshold=None):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]

        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            data = data[group_indices]

        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        stress = data[:, 5]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if percentile_threshold is not None:
            threshold = np.percentile(stress, percentile_threshold)
            high_stress_mask = stress >= threshold
            ax.scatter(x, y, z, c='lightgray', s=5, alpha=0.1)
            scatter = ax.scatter(x[high_stress_mask], y[high_stress_mask], z[high_stress_mask],
                        c=stress[high_stress_mask], cmap='hot', s=30, alpha=1.0)
            title = f'High Stress Regions (>{percentile_threshold}%) - Timestep {current_timestep}'
        else:
            scatter = ax.scatter(x, y, z, c=stress, cmap='hot', s=10, alpha=0.7)
            title = f'3D von Mises Stress Distribution - Timestep {current_timestep}'
        
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        
    
        plt.colorbar(scatter, ax=ax, label='von Mises Stress')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f'stress_3d_timestep_{current_timestep}.png', dpi=300)

    def plot_stress_by_layer(self, timestep_idx=-1, axis='z', layers_to_create=10):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx

        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]

        atoms_spatial_coordinates = self.parser.get_atoms_spatial_coordinates(data)
        coords = get_data_from_coord_axis(axis, atoms_spatial_coordinates)
        axis_name = axis.upper()
        stress = data[:, 5]
        
        min_coord = np.min(coords)
        max_coord = np.max(coords)
        layer_edges = np.linspace(min_coord, max_coord, layers_to_create + 1)

        layer_centers = []
        average_stress_by_layer = []
        max_stress_by_layer = []
        min_stress_by_layer = []
        atoms_in_layer = []

        for i in range(layers_to_create):
            layer_min = layer_edges[i]
            layer_max = layer_edges[i + 1]
            layer_center = (layer_min + layer_max) / 2

            layer_mask = (coords >= layer_min) & (coords < layer_max)
            layer_stress = stress[layer_mask]

            layer_centers.append(layer_center)
            atoms_in_layer.append(np.sum(layer_mask))

            if len(layer_stress) > 0:
                average_stress_by_layer.append(np.mean(layer_stress))
                max_stress_by_layer.append(np.max(layer_stress))
                min_stress_by_layer.append(np.min(layer_stress))
            else:
                average_stress_by_layer.append(0)
                max_stress_by_layer.append(0)
                min_stress_by_layer.append(0)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            ax1.plot(layer_centers, average_stress_by_layer, 'b-o', label='Average Stress')
            ax1.plot(layer_centers, max_stress_by_layer, 'r-^', label='Maximum Stress')
            ax1.fill_between(layer_centers, min_stress_by_layer, max_stress_by_layer, color='blue', alpha=0.2)

            ax1.set_xlabel(f'Coordinate {axis_name} (Å)')
            ax1.set_ylabel(f'von Mises Stress')
            ax1.set_title(f'Stress by Layer - Axis {axis_name} (Timestep {current_timestep})')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()

            ax2.bar(layer_centers, atoms_in_layer, width=(layer_edges[1] - layer_edges[0]) * 0.8, alpha=0.7)
            ax2.set_xlabel(f'Coordinate {axis_name} (Å)')
            ax2.set_ylabel('Number of Atoms')
            ax2.set_title('Distribution of Atoms per Layer')
            ax2.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(f'stress_by_layer_{axis}_timestep_{current_timestep}.png', dpi=300)