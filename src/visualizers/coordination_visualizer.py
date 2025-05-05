from analyzers.coordination_analyzer import CoordinationAnalyzer
from core.base_parser import BaseParser
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

class CoordinationVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = CoordinationAnalyzer(parser)
        # TODO: move this to analyzers/coordination
        self.coord_colors = {
            # Low coordination (1-4)
            'low': 'red',
            # Surface atoms (5-8)
            'surface': 'orange',
            # Defect (9-11)
            'defect': 'yellow',
            # Perfect (12)
            'perfect': 'green',
            # Excess (13+)
            'excess': 'blue'
        }

    def plot_coord_distribution(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        coord_values, counts, percentages = self.analyzer.get_coord_distribution(timestep_idx)
        plt.figure(figsize=(12, 7))
        bars = plt.bar(coord_values, percentages, alpha=0.7)
        for i, bar in enumerate(bars):
            value = coord_values[i]
            if value < 5:
                bar.set_color(self.coord_colors['low'])
            elif value < 9:
                bar.set_color(self.coord_colors['surface'])
            elif value < 12:
                bar.set_color(self.coord_colors['defect'])
            elif value == 12:
                bar.set_color(self.coord_colors['perfect'])
            else:
                bar.set_color(self.coord_colors['excess'])
        plt.xlabel('Coordination Number')
        plt.ylabel('Atoms Percentage (%)')
        plt.title(f'Distribution of Coordination Numbers (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(coord_values)
        stats = self.analyzer.get_coord_stats(timestep_idx)
        stats_text = (f'Average Coordination {stats["mean"]:.2f}\n'
                      f'Perfect Atoms (12): {stats["perfect_ratio"]:.1f}%\n'
                      f'Atoms with Defects: {stats["defect_ratio"]:.1f}%')
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        legend_elements = [
            Patch(facecolor=self.coord_colors['low'], label='Very low (1-4)'),
            Patch(facecolor=self.coord_colors['surface'], label='Surface (5-8)'),
            Patch(facecolor=self.coord_colors['defect'], label='Minor defects (9-11)'),
            Patch(facecolor=self.coord_colors['perfect'], label='Pefect (12)'),
            Patch(facecolor=self.coord_colors['excess'], label='Excessive (13+)')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'coordination_distribution_timestep_{current_timestep}.png', dpi=300)
        plt.close()

    def plot_coord_evolution(self):
        timesteps, mean_coord, perfect_ratio, defect_ratio = self.analyzer.get_coord_evolution()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(timesteps, mean_coord, 'b-', linewidth=2)
        ax1.set_ylabel('Average Coordination')
        ax1.set_title('Evolution of Atomic Coordination')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.axhline(y=12, color='green', linestyle='--', alpha=0.7, label='Perfect Coordination (12)')
        ax1.legend()

        ax2.plot(timesteps, perfect_ratio, 'g-', label='Perfect Atoms (12)', linewidth=2)
        ax2.plot(timesteps, defect_ratio, 'r-', label='Atoms with Defects', linewidth=2)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Percentage (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('coordination_evolution.png', dpi=300)
        plt.close()

    def plot_coord_spatial(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        current_timestep = timesteps[timestep_idx]
        x, y, z, coord = self.analyzer.get_spatial_distribution(timestep_idx)
        cmap_colors = [
            'red', 'red', 'red', 'red', 'red', 
            'orange', 'orange', 'orange', 'orange',
            'yellow', 'yellow', 'yellow',
            'green',
            'blue', 'blue', 'blue'
        ]
        coord_cmap = ListedColormap(cmap_colors)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        sc1 = axs[0].scatter(x, y, c=coord, cmap=coord_cmap, s=5, alpha=0.8, vmin=0, vmax=15)
        axs[0].set_title('Coordination - XY Plane (Top View)')
        axs[0].set_xlabel('X (Å)')
        axs[0].set_ylabel('Y (Å)')
        
        sc2 = axs[1].scatter(x, z, c=coord, cmap=coord_cmap, s=5, alpha=0.8, vmin=0, vmax=15)
        axs[1].set_title('Coordination - XZ Plane (Side View)')
        axs[1].set_xlabel('X (Å)')
        axs[1].set_ylabel('Z (Å)')
        
        sc3 = axs[2].scatter(y, z, c=coord, cmap=coord_cmap, s=5, alpha=0.8, vmin=0, vmax=15)
        axs[2].set_title('Coordination - YZ Plane (Front View)')
        axs[2].set_xlabel('Y (Å)')
        axs[2].set_ylabel('Z (Å)')
        
        cbar = fig.colorbar(sc1, ax=axs, ticks=range(16))
        cbar.set_label('Coordination Number')
        
        plt.suptitle(f'Coordination Spatial Distribution (Timestep {current_timestep})', y=1.05)
        plt.tight_layout()
        plt.savefig(f'coordination_spatial_timestep_{current_timestep}.png', dpi=300)
        plt.close()

    def plot_atom_classification(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        perfect, surface, defect = self.analyzer.classify_atoms(timestep_idx)
        x, y, z, _ = self.analyzer.get_spatial_distribution(timestep_idx)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if len(perfect) > 0:
            ax.scatter(x[perfect], y[perfect], z[perfect], s=5, alpha=0.2, c='green', label='Perfect')
        
        if len(defect) > 0:
            ax.scatter(x[defect], y[defect], z[defect], s=20, alpha=0.8, c='yellow', label='Minor Defects')
        
        if len(surface) > 0:
            ax.scatter(x[surface], y[surface], z[surface], s=20, alpha=0.8, c='red', label='Surface/Serious Defects')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Classification of Atoms by Coordination (Timestep {current_timestep})')
        
        total = len(x)
        perfect_pct = len(perfect) / total * 100
        surface_pct = len(surface) / total * 100
        defect_pct = len(defect) / total * 100

        stats = (f'Perfect Atoms: {perfect_pct:.1f}%\n'
                f'Surface Atoms: {surface_pct:.1f}%\n'
                f'Minor Defect Atoms: {defect_pct:.1f}%')
        
        ax.text2D(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'atom_classification_timestep_{current_timestep}.png', dpi=300)
        plt.close()

    def plot_coord_ranges(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        current_timestep = timesteps[timestep_idx]
        ranges, counts, percentages = self.analyzer.get_coord_range_distribution(timestep_idx)
        colors = [self.coord_colors['low'], self.coord_colors['surface'], 
                self.coord_colors['defect'], self.coord_colors['perfect'], 
                self.coord_colors['excess']]
        plt.figure(figsize=(10, 7))
        bars = plt.bar(ranges, percentages, color=colors, alpha=0.7)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentages[i]:.1f}%', ha='center', va='bottom', rotation=0)
        plt.xlabel('Coordination Range')
        plt.ylabel('Percentage of Atoms (%)')
        plt.title(f'Distribution of Atoms by Coordination Range (Timestep {current_timestep})')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        interpretations = [
            'Very low: Atoms in corners or very disordered',
            'Surface: Atoms on surfaces or edges',
            'Minor defects: Imperfect structure',
            'Perfect: Ideal FCC structure',
            'Excessive: Compression or overlap'
        ]

        plt.text(0.5, -0.15, '\n'.join(interpretations), transform=plt.gca().transAxes,
            ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'coordination_ranges_timestep_{current_timestep}.png', dpi=300)
        plt.close()

    def plot_coord_comparison(self, timestep_idx1=0, timestep_idx2=-1):
        comparison = self.analyzer.compare_timesteps(timestep_idx1, timestep_idx2)
        ranges = comparison['ranges']
        percentages1 = comparison['percentages1']
        percentages2 = comparison['percentages2']
        timestep1 = comparison['timestep1']
        timestep2 = comparison['timestep2']
        x = np.arange(len(ranges))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars1 = ax.bar(x - width/2, percentages1, width, label=f'Timestep {timestep1}', alpha=0.7)
        bars2 = ax.bar(x + width/2, percentages2, width, label=f'Timestep {timestep2}', alpha=0.7)
        
        ax.set_xlabel('Coordination Range')
        ax.set_ylabel('Percentage of Atoms (%)')
        ax.set_title('Coordination Comparison between Timesteps')
        ax.set_xticks(x)
        ax.set_xticklabels(ranges)
        ax.legend()
        
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        def autolabel(bars, values):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.annotate(
                    f'{values[i]:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', 
                    va='bottom')
                
        autolabel(bars1, percentages1)
        autolabel(bars2, percentages2)
        
        differences = [p2 - p1 for p1, p2 in zip(percentages1, percentages2)]
        
        def get_change_summary():
            summary = []
            if differences[3] < 0:
                summary.append(f'Decrease of {abs(differences[3]):.1f}% in perfect atoms')
            if differences[0] > 0 or differences[1] > 0:
                summary.append(f'Increase of {differences[0] + differences[1]:.1f}% in serious defects/surfaces')
            if differences[2] > 0:
                summary.append(f'{differences[2]:.1f}% increase in minor defects')
            return '\n'.join(summary)

        change_summary = get_change_summary()
        if change_summary:
            plt.text(0.5, -0.15, f"Cambios principales:\n{change_summary}", transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        plt.tight_layout()
        plt.savefig(f'coordination_comparison_{timestep1}_vs_{timestep2}.png', dpi=300)
        plt.close()