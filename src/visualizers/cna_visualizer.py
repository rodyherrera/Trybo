from analyzers.cna_analyzer import CommonNeighborAnalysisAnalyzer
from core.base_parser import BaseParser
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

class CommonNeighborAnalysisVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = CommonNeighborAnalysisAnalyzer(parser)

        # TODO: move to analyzers/cna_analyzer
        self.structure_colors = {
            0: 'gray',
            1: 'green',
            2: 'blue',
            3: 'red',
            4: 'purple',
            5: 'orange'
        }
    
    def plot_structure_distribution(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        structure_counts = self.analyzer.get_structure_counts(timestep_idx)
        current_timestep = timesteps[timestep_idx]

        labels = []
        sizes = []
        colors = []

        for struct_type in sorted(structure_counts.keys()):
            label = self.analyzer.structure_names.get(struct_type, f'Type {struct_type}')
            color = self.structure_colors.get(struct_type, 'lightgray')
            labels.append(f'{label} ({structure_counts[struct_type]})')
            sizes.append(structure_counts[struct_type])
            colors.append(color)
        
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title(f'Crystal Structure Distribution (Timestep {current_timestep})')
        plt.tight_layout()
        plt.savefig(f'cna_distribution_timestep_{current_timestep}.png', dpi=300)
        plt.close()

    def plot_structure_evolution(self):
        timesteps = self.parser.get_timesteps()
        evolution = self.analyzer.get_structure_evolution()
        plt.figure(figsize=(12, 8))

        for struct_type, percentages in evolution.items():
            label = self.analyzer.structure_names.get(struct_type, f'Type {struct_type}')
            color = self.structure_colors.get(struct_type, 'lightgray')
            plt.plot(timesteps, percentages, label=label, color=color, linewidth=2)
        
        plt.xlabel('Timestep')
        plt.ylabel('Structure Percentage (%)')
        plt.title('Evolution of Crystal Structures Over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('structure_evolution.png', dpi=300)
        plt.close()

    def plot_structure_heatmap(self, timestep_idx=-1):
        timesteps = self.parser.get_timesteps()
        current_timestep = timesteps[timestep_idx]
        x, y, z, cna = self.analyzer.get_spatial_distribution(timestep_idx)
        cmap_colors = [self.structure_colors.get(i, 'lightgray') for i in range(6)]
        cmap = ListedColormap(cmap_colors)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        scatter_xy = axs[0].scatter(x, y, c=cna, cmap=cmap, s=5, alpha=0.8, vmin=0, vmax=5)
        axs[0].set_title('Crystal Structure - XY Plane (Top View)')
        axs[0].set_xlabel('X (Å)')
        axs[0].set_ylabel('Y (Å)')

        scatter_xz = axs[1].scatter(x, z, c=cna, cmap=cmap, s=5, alpha=0.8, vmin=0, vmax=5)
        axs[1].set_title('Crystal Structure - XZ Plane (Side View)')
        axs[1].set_xlabel('X (Å)')
        axs[1].set_ylabel('Z (Å)')

        scatter_yz = axs[2].scatter(y, z, c=cna, cmap=cmap, s=5, alpha=0.8, vmin=0, vmax=5)
        axs[2].set_title('Crystal Structure - YZ Plane (Front View)')
        axs[2].set_xlabel('Y (Å)')
        axs[2].set_ylabel('Z (Å)')

        cbar = fig.colorbar(scatter_xy, ax=axs, ticks=[0, 1, 2, 3, 4, 5])
        cbar.set_label('Crystal Structure')
        cbar.set_ticklabels(list(self.analyzer.structure_names.values()))
        
        plt.suptitle(f'Crystal Structure Distribution - Timestep {current_timestep}', y=1.05)
        plt.tight_layout()
        plt.savefig(f'cna_spatial_distribution_timestep_{current_timestep}.png', dpi=300)
        plt.close()

    def plot_structure_comparison(self, timestep_idx1=0, timestep_idx2=-1):
        timesteps = self.parser.get_timesteps()
        comparison = self.analyzer.compare_structures(timestep_idx1, timestep_idx2)
        timestep1 = timesteps[timestep_idx1]
        timestep2 = timesteps[timestep_idx2]

        x = np.arange(len(comparison['names']))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, comparison['percentages1'], width, label=f'Timestep {timestep1}')
        rects2 = ax.bar(x + width/2, comparison['percentages2'], width, label=f'Timestep {timestep2}')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Crystal Structure Comparison Between Timesteps')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['names'])
        ax.legend()

        # Add value annotations on top of bars
        def autolabel(rects, percentages):
            for rect, pct in zip(rects, percentages):
                height = rect.get_height()
                ax.annotate(f'{pct:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom')
        
        autolabel(rects1, comparison['percentages1'])
        autolabel(rects2, comparison['percentages2'])
        
        fig.tight_layout()
        plt.savefig(f'cna_comparison_timestep_{timestep1}_vs_{timestep2}.png', dpi=300)
        plt.close()