from analyzers.hotspot_analyzer import HotspotAnalyzer
from core.base_parser import BaseParser
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class HotspotVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = HotspotAnalyzer(parser)
    
    def plot_energy_distribution(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        energy_bins, _, histogram_normalized, threshold = self.analyzer.get_energy_distribution(timestep_idx)
        stats = self.analyzer.get_hotspot_stats(timestep_idx)
        plt.figure(figsize=(12, 7))
        bin_centers = (energy_bins[::-1] + energy_bins[1:]) / 2
        plt.bar(bin_centers, histogram_normalized, width=bin_centers[1] - bin_centers[0], alpha=0.7)
        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'Hotspot Threshold: {threshold:.2e}')
            plt.fill_between([threshold, max(energy_bins)], [0, 0], [100, 100], color='red', alpha=0.1)
        plt.xlabel('Kinetic Energy (eV)')
        plt.ylabel('Percentage of Atoms (%)')
        plt.title(f'Kinetic Energy Distribution (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        if max(energy_bins) / (min(energy_bins) + 1e-10) > 50:
            plt.xscale('log')

        stats_text = (f'Total atoms: {stats["total_atoms"]}\n'
            f'Hotspot atoms: {stats["hotspot_count"]} ({stats["hotspot_ratio"]:.2f}%)\n'
            f'Average energy: {stats["avg_energy"]:.2e} eV\n'
            f'Hotspot avg energy: {stats["hotspot_avg_energy"]:.2e} eV\n'
            f'Maximum energy: {stats["max_energy"]:.2e} eV')

        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'energy_distribution_timestep_{current_timestep}.png', dpi=300)
        plt.close()
    
    def plot_hotspot_evolution(self):
        timesteps, hotspot_counts, hotspot_ratios, average_energies, max_energies = self.analyzer.get_hotspot_evolution()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        ax1.plot(timesteps, hotspot_counts, 'r-', linewidth=2)
        ax1.set_ylabel('Number of Hotspots')
        ax1.set_title('Evolution of Thermal Hotspots')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(timesteps, hotspot_ratios, 'b--', linewidth=2)
        ax1_twin.set_ylabel('Hotspot Ratio (%)', color='b')
        ax1_twin.tick_params(axis='y', labelcolor='b')

        ax2.plot(timesteps, average_energies, 'g-', linewidth=2)
        ax2.set_ylabel('Average Kinetic Energy (eV)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        ax3.plot(timesteps, max_energies, 'purple', linewidth=2)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Maximum Kinetic Energy (eV)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('hotspot_evolution.png', dpi=300)
        plt.close()

    def plot_hotspot_spatial(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        current_timestep = timesteps[timestep_idx]
        x, y, z, ke_values, is_hotspot = self.analyzer.get_spatial_energy_distribution(timestep_idx)
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        cmap = plt.cm.plasma

        sc1 = axs[0, 0].scatter(x, y, c=ke_values, cmap=cmap, s=5, alpha=0.7)
        axs[0, 0].set_title('Energy Distribution - XY Plane')
        axs[0, 0].set_xlabel('X (Å)')
        axs[0, 0].set_ylabel('Y (Å)')

        sc2 = axs[0, 1].scatter(x, z, c=ke_values, cmap=cmap, s=5, alpha=0.7)
        axs[0, 1].set_title('Energy Distribution - XZ Plane')
        axs[0, 1].set_xlabel('X (Å)')
        axs[0, 1].set_ylabel('Z (Å)')
        
        sc3 = axs[0, 2].scatter(y, z, c=ke_values, cmap=cmap, s=5, alpha=0.7)
        axs[0, 2].set_title('Energy Distribution - YZ Plane')
        axs[0, 2].set_xlabel('Y (Å)')
        axs[0, 2].set_ylabel('Z (Å)')

        cbar = fig.colorbar(sc1, ax=axs[0, :])
        cbar.set_label('Kinetic Energy (eV)')

        hotspot_mask = is_hotspot > 0
        axs[1, 0].scatter(x, y, color='lightgray', s=2, alpha=0.3)
        axs[1, 1].scatter(x, z, color='lightgray', s=2, alpha=0.3)
        axs[1, 2].scatter(y, z, color='lightgray', s=2, alpha=0.3)

        if np.any(hotspot_mask):
            axs[1, 0].scatter(x[hotspot_mask], y[hotspot_mask], color='red', s=20, alpha=0.8)
            axs[1, 1].scatter(x[hotspot_mask], z[hotspot_mask], color='red', s=20, alpha=0.8)
            axs[1, 2].scatter(y[hotspot_mask], z[hotspot_mask], color='red', s=20, alpha=0.8)
            
        axs[1, 0].set_title('Hotspot Locations - XY Plane')
        axs[1, 0].set_xlabel('X (Å)')
        axs[1, 0].set_ylabel('Y (Å)')
        
        axs[1, 1].set_title('Hotspot Locations - XZ Plane')
        axs[1, 1].set_xlabel('X (Å)')
        axs[1, 1].set_ylabel('Z (Å)')
        
        axs[1, 2].set_title('Hotspot Locations - YZ Plane')
        axs[1, 2].set_xlabel('Y (Å)')
        axs[1, 2].set_ylabel('Z (Å)')
        
        plt.suptitle(f'Spatial Distribution of Energy and Hotspots (Timestep {current_timestep})', y=1.05, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'hotspot_spatial_distribution_timestep_{current_timestep}.png', dpi=300)
        plt.close()
    
    def plot_hotspot_clusters_3d(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        current_timestep = timesteps[timestep_idx]
        x, y, z, ke_values, is_hotspot = self.analyzer.get_spatial_distribution(timestep_idx)
        clusters, cluster_sizes, cluster_positions = self.analyzer.get_hotspot_clusters(timestep_idx)
        if not clusters:
            print(f'No hotspot clusters found in timestep {current_timestep}')
            return
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        background_mask = is_hotspot == 0
        ax.scatter(x[background_mask], y[background_mask], z[background_mask], color='lightgray', s=1, alpha=0.1)
        colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
        for i, (cluster_id, atom_indices) in enumerate(clusters.items()):
            color = colors[i % len(colors)]
            ax.scatter(x[atom_indices], y[atom_indices], z[atom_indices], color=color, s=30, alpha=0.8, label=f'Cluster {cluster_id} ({cluster_sizes[cluster_id]} atoms)')
            if cluster_sizes[cluster_id] > 1:
                cx, cy, cz = cluster_positions[cluster_id]
                ax.scatter([cx], [cy], [cz], color='black', s=80, marker='*')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Hotspot Clusters (Timestep {current_timestep})')

        if len(clusters) <= 10:
            ax.lengend()
        else:
            ax.text2D(0.05, 0.95, f'Total clusters: {len(clusters)}', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'hotspot_clusters_3d_timestep_{current_timestep}.png', dpi=300)
        plt.close()
    
    def plot_hotspot_heatmap(self, timestep_idx=-1, bins=50):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        x, y, z, ke_values, is_hotspot = self.analyzer.get_spatial_distribution(timestep_idx)
        hotspot_mask = is_hotspot > 0
       
        if not np.any(hotspot_mask):
            print(f'No hotspots found in timestep {current_timestep}')
            return

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        hxy = axs[0].hist2d(x[hotspot_mask], y[hotspot_mask], bins=bins, cmap='hot')
        axs[0].set_title('Hotspot Concentration - XY Plane')
        axs[0].set_xlabel('X (Å)')
        axs[0].set_ylabel('Y (Å)')
        plt.colorbar(hxy[3], ax=axs[0], label='Hotspot Count')
        
        hxz = axs[1].hist2d(x[hotspot_mask], z[hotspot_mask], bins=bins, cmap='hot')
        axs[1].set_title('Hotspot Concentration - XZ Plane')
        axs[1].set_xlabel('X (Å)')
        axs[1].set_ylabel('Z (Å)')
        plt.colorbar(hxz[3], ax=axs[1], label='Hotspot Count')
        
        hyz = axs[2].hist2d(y[hotspot_mask], z[hotspot_mask], bins=bins, cmap='hot')
        axs[2].set_title('Hotspot Concentration - YZ Plane')
        axs[2].set_xlabel('Y (Å)')
        axs[2].set_ylabel('Z (Å)')
        plt.colorbar(hyz[3], ax=axs[2], label='Hotspot Count')
        
        plt.suptitle(f'Hotspot Concentration Heatmaps (Timestep {current_timestep})', y=1.05)
        plt.tight_layout()
        plt.savefig(f'hotspot_heatmap_timestep_{current_timestep}.png', dpi=300)
        plt.close()