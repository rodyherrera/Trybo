from analyzers.debris_analyzer import DebrisAnalyzer
from core.base_parser import BaseParser
import matplotlib.pyplot as plt
import numpy as np

class DebrisVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = DebrisAnalyzer(parser)

    def plot_cluster_evolution(self):
        timesteps, num_clusters, largest_cluster, average_cluster = self.analyzer.get_cluster_evolution()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(timesteps, num_clusters, 'b--', linewidth=2)
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Evolution of Debris Clusters')
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(timesteps, largest_cluster, 'r-', label='Largest Cluster', linewidth=2)
        ax2.plot(timesteps, average_cluster, 'g--', label='Average Size', linewidth=2)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Cluster Size (atoms)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('debris_cluster_evolution.png', dpi=300)
        plt.close()

    def plot_cluster_size_distribution(self, timestep_idx=-1, min_size=2, log_scale=True):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        sizes, counts = self.analyzer.get_cluster_size_distribution(timestep_idx, min_size)

        if not sizes:
            print(f'No clusters of size >= {min_size} were found at timestep {current_timestep}')
            return

        plt.figure(figsize=(10, 6))
        bars = plt.bar(sizes, counts, alpha=0.7, color='steelblue')
        plt.xlabel('Cluster Size (number of atoms)')
        plt.ylabel('Frequency')
        plt.title(f'Debris Cluster Size Distribution (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        if log_scale and max(sizes) > min_size * 5:
            plt.xscale('log')

        if sizes:
            stats = (f'Total Clusters: {sum(counts)}\n'
                    f'Average Size: {np.average(sizes, weights=counts):.1f} atoms\n'
                    f'Largest Cluster: {max(sizes)} atoms\n'
                    f'Smallest Cluster: {min(sizes)} atoms')
                
            plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, verticalalignment='top', 
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'debris_size_distribution_timestep_{current_timestep}.png', dpi=300)
        plt.close()

    def plot_3d_cluster_visualization(self, timestep_idx=-1, min_size=5, max_clusters=10):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx

        current_timestep = timesteps[timestep_idx]
        cluster_positions, cluster_sizes, atom_coords = self.analyzer.get_cluster_spatial_data(timestep_idx, min_size)
        if not cluster_positions:
            print(f'No clusters of size >= {min_size} were found at timestep {current_timestep}')
            return
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:max_clusters]
        cluster_ids = [cluster[0] for cluster in top_clusters]
        colors = plt.cm.tab10(np.linspace(0, 1, max_clusters))
        color_dict = {cluster_id: colors[i % max_clusters] for i, cluster_id in enumerate(cluster_ids)}
        x = atom_coords['x']
        y = atom_coords['y']
        z = atom_coords['z']
        cluster_ids_all = atom_coords['cluster_id']
        mask_others = ~np.isin(cluster_ids_all, cluster_ids)
        ax.scatter(x[mask_others], y[mask_others], z[mask_others], s=10, alpha=0.1, c='gray', label='Others')
        for cluster_id in cluster_ids:
            mask = cluster_ids_all == cluster_id
            if np.any(mask):
                ax.scatter(x[mask], y[mask], z[mask], s=30, c=[color_dict[cluster_id]], label=f'Cluster {int(cluster_id)} ({cluster_sizes[cluster_id]} atoms)')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'3D Visualization of Debris Clusters (Timestep {current_timestep})')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(f'debris_3d_visualization_timestep_{current_timestep}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_2d_projections(self, timestep_idx=-1, min_size=5, max_clusters=10):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        cluster_positions, cluster_sizes, atom_coords = self.analyzer.get_cluster_spatial_data(timestep_idx, min_size)
        if not cluster_positions:
            print(f'No clusters of size >= {min_size} were found at timestep {current_timestep}')
            return
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:max_clusters]
        cluster_ids = [cluster[0] for cluster in top_clusters]
        colors = plt.cm.tab10(np.linspace(0, 1, max_clusters))
        color_dict = {cluster_id: colors[i % max_clusters] for i, cluster_id in enumerate(cluster_ids)}
        x = atom_coords['x']
        y = atom_coords['y']
        z = atom_coords['z']
        cluster_ids_all = atom_coords['cluster_id']
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        def plot_projection(ax, x_data, y_data, title):
            mask_others = ~np.isin(cluster_ids_all, cluster_ids)
            ax.scatter(x_data[mask_others], y_data[mask_others], s=5, alpha=0.1, c='gray')
            for cluster_id in cluster_ids:
                mask = cluster_ids_all == cluster_id
                if np.any(mask):
                    ax.scatter(x_data[mask], y_data[mask], s=20, c=[color_dict[cluster_id]], label=f'Cluster {int(cluster_id)} ({cluster_sizes[cluster_id]} atoms)')
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.3)
        
        plot_projection(axs[0], x, y, 'XY Plane (Top View)')
        axs[0].set_xlabel('X (Å)')
        axs[0].set_ylabel('Y (Å)')
        
        plot_projection(axs[1], x, z, 'XZ Plane (Side View)')
        axs[1].set_xlabel('X (Å)')
        axs[1].set_ylabel('Z (Å)')
        
        plot_projection(axs[2], y, z, 'YZ Plane (Front View)')
        axs[2].set_xlabel('Y (Å)')
        axs[2].set_ylabel('Z (Å)')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=min(5, len(labels)))
        plt.suptitle(f'2D Projections of Debris Clusters (Timestep {current_timestep})', y=1.05)
        plt.tight_layout()
        plt.savefig(f'debris_2d_projections_timestep_{current_timestep}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_largest_clusters_info(self, timestep_idx=-1, n=5):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        top_clusters = self.analyzer.get_largest_clusters(timestep_idx, n)
        if not top_clusters:
            print(f'No clusters were found at timestep {current_timestep}')
            return

        plt.figure(figsize=(8, max(5, len(top_clusters) * 0.5 + 1)))
        plt.axis('tight')
        plt.axis('off')

        table_data = []
        for i, (cluster_id, size) in enumerate(top_clusters):
            table_data.append([ i + 1, int(cluster_id), size ])
        
        table = plt.table(
            cellText=table_data,
            colLabels=['Range', 'Cluster ID', 'Size (atoms)'],
            loc='center',
            cellLoc='center',
            colWidths=[0.2, 0.4, 0.4]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        plt.title(f'The {len(top_clusters)} Largest Clusters (Timestep {current_timestep})', y=0.9, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'largest_clusters_info_timestep_{current_timestep}.png', dpi=300, bbox_inches='tight')
        plt.close()