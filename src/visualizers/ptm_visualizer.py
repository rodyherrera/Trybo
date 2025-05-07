from analyzers.ptm_analyzer import PTMAnalyzer
from core.base_parser import BaseParser
from utilities.analyzer import get_coords, get_data_from_coord_axis
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class PTMVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = PTMAnalyzer(parser)

    def plot_structure_distribution(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        current_timestep = timesteps[timestep_idx]
        counts = self.analyzer.get_structure_distribution(timestep_idx, group)
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        structure_names = [item[0] for item in sorted_items]
        structure_counts = [item[1] for item in sorted_items]
        colors = []
        for name in structure_names:
            for key, value in self.analyzer.structure_names.items():
                if value == name:
                    colors.append(self.analyzer.structure_colors[key])
                    break
            else:
                colors.append('gray')
        plt.figure(figsize=(12, 8))
        bars = plt.bar(structure_names, structure_counts, color=colors)
        total = sum(structure_counts)
        for bar, count in zip(bars, structure_counts):
            percentage = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{percentage:.1f}%', ha='center', va='bottom')
        plt.xlabel('Crystal Structure')
        plt.ylabel('Atoms Number')
        title = f'Crystal Structure Distribution (Timestep {current_timestep})'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(f'structure_distribution_timestep_{current_timestep}.png', dpi=300)
    
    def plot_structure_evolution(self, group=None):
        timesteps, evolution = self.analyzer.get_structure_evolution(group)
        plt.figure(figsize=(12, 8))
        for name, counts in evolution.items():
            for key, value in self.analyzer.structure_names.items():
                if value == name:
                    color = self.analyzer.structure_colors[key]
                    break
            else:
                color = 'gray'
            plt.plot(timesteps, counts, marker='o', linestyle='-', label=name, color=color)
        plt.xlabel('Timestep')
        plt.ylabel('Atoms Number')
        title = 'Evolution of Crystal Structures'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('structure_evolution.png', dpi=300)
    
    def plot_3d_structures(self, timestep_idx=-1, group=None, filter_rmsd=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            data = data[group_indices]
        x, y, z = get_coords(data)
        # c_ptm[1]
        structure_types = data[:, 5].astype(int)
        # c_ptm[2]
        rmsd_values = data[:, 6]
        # filter by rmsd if specified
        if filter_rmsd is not None:
            valid_indices = ~np.isinf(rmsd_values) & (rmsd_values <= filter_rmsd)
            x = x[valid_indices]
            y = y[valid_indices]
            z = z[valid_indices]
            structure_types = structure_types[valid_indices]
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for struct_type, name in self.analyzer.structure_names.items():
            mask = structure_types == struct_type
            if np.any(mask):
                color = self.analyzer.structure_colors[struct_type]
                ax.scatter(x[mask], y[mask], z[mask], c=color, s=10, alpha=0.7, label=name)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        title = f'3D Distribution of Crystal Structures (Timestep {current_timestep})'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        if filter_rmsd is not None:
            title += f' - RMSD < {filter_rmsd}'
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'structures_3d_timestep_{current_timestep}.png', dpi=300)

    def plot_rmsd_distribution(self, timestep_idx=-1, group=None, max_rmsd=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            data = data[group_indices]
        rmsd_values = data[:, 6]
        valid_rmsd = rmsd_values[~np.isinf(rmsd_values) & ~np.isnan(rmsd_values)]
        if max_rmsd is not None:
            valid_rmsd = valid_rmsd[valid_rmsd <= max_rmsd]
        plt.figure(figsize=(10, 8))
        if len(valid_rmsd) > 0:
            sns.histplot(valid_rmsd, kde=True, bins=50)
            plt.axvline(np.mean(valid_rmsd), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_rmsd):.5f}')
            plt.axvline(np.median(valid_rmsd), color='blue', linestyle='-.', 
                       label=f'Median: {np.median(valid_rmsd):.5f}')
            stats = (f'Mean: {np.mean(valid_rmsd):.5f}\n'
                    f'Median: {np.median(valid_rmsd):.5f}\n'
                    f'Max: {np.max(valid_rmsd):.5f}\n'
                    f'Min: {np.min(valid_rmsd):.5f}\n'
                    f'STD. Est.: {np.std(valid_rmsd):.5f}')
            plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, 
                   verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            plt.text(0.5, 0.5, 'There are no valid RMSD values', transform=plt.gca().transAxes, ha='center')
        plt.xlabel('RMSD')
        plt.ylabel('Frequency')
        title = f'RMSD distribution (Timestep {current_timestep})'
        if group is not None and group != 'all':
            title += f' - Group: {group}'
        if max_rmsd is not None:
            title += f' - Max RMSD: {max_rmsd}'
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'rmsd_distribution_timestep_{current_timestep}.png', dpi=300)
    
    def plot_structure_by_layer(self, timestep_idx=-1, axis='z', n_layers=10):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]
        coords = get_data_from_coord_axis(axis, data)
        axis_name = axis.upper()
        structure_types = data[:, 5].astype(int)
        min_coord = np.min(coords)
        max_coord = np.max(coords)
        layer_edges = np.linspace(min_coord, max_coord, n_layers + 1)
        layer_centers = [(layer_edges[i] + layer_edges[i + 1]) / 2 for i in range(n_layers)]
        structures_by_layer = []
        for i in range(n_layers):
            layer_min = layer_edges[i]
            layer_max = layer_edges[i + 1]
            layer_mask = (coords >= layer_min) & (coords < layer_max)
            layer_structures = structure_types[layer_mask]
            layer_counts = {}
            for struct_type, name in self.analyzer.structure_names.items():
                layer_counts[name] = np.sum(layer_structures == struct_type)
            structures_by_layer.append(layer_counts)
        plt.figure(figsize=(12, 8))
        bottoms = np.zeros(n_layers)
        for name in self.analyzer.structure_names.values():
            counts = [layer_counts[name] for layer_counts in structures_by_layer]
            for key, value in self.analyzer.structure_names.items():
                if value == name:
                    color = self.analyzer.structure_colors[key]
                    break
            else:
                color = 'gray'
            plt.bar(layer_centers, counts, bottom=bottoms, label=name, color=color, width=(layer_edges[1] - layer_edges[0]) * 0.8)
            bottoms += counts
        plt.xlabel(f'Coordinate {axis_name} (Å)')
        plt.ylabel('Atoms Number')
        plt.title(f'Distribution of Structures by Layer - Axis {axis_name} (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'structures_by_layer_{axis}_timestep_{current_timestep}.png', dpi=300)
    
    def create_ptm_animation(self, interval=200):
        timesteps = self.parser.get_timesteps()
        if len(timesteps) < 2:
            print('At least 2 timesteps are needed to create an animation')
            return
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            data = self.parser.get_data()[frame]
            current_timestep = timesteps[frame]
            x, y, z = get_coords(data)
            structure_types = data[:, 5].astype(int)
            for struct_type, name in self.analyzer.structure_names.items():
                mask = struct_type == struct_type
                if np.any(mask):
                    color = self.analyzer.structure_colors[struct_type]
                    ax.scatter(x[mask], y[mask], z[mask], c=color, s=10, alpha=0.7, label=name)
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(f'Crystal Structures (Timestep {current_timestep})')
            ax.legend()
            return ax,

        anim = animation.FuncAnimation(fig, update, frames=len(timesteps), interval=interval, blit=False)
        try:
            anim.save('ptm_animation.gif', writer='pillow', fps=1000/interval)
        except Exception as e:
            print(f'Error saving animation: {e}')
        plt.close(fig)