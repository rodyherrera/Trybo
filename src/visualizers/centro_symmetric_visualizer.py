from core.base_parser import BaseParser
from analyzers.centro_symmetric_analyzer import CentroSymmetricAnalyzer
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
    
    def plot_centro_symmetric_distribution(self, timestep_idx=-1, group=None, log_sacle=False):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]
        current_timestep = timesteps[timestep_idx]

        if group is not None and group != 'all':
            group_indices = self.analyzer.get_atom_group_indices()[group]
            data = data[group_indices]

        centro_symmetric_values = data[:, 5]
        plt.figure(figsize=(12, 8))
        ax = sns.histplot(centro_symmetric_values, kde=True, bins=50)
        # Vertical lines for classification thresholds
        for struct_type, (min_value, max_value) in self.analyzer.structure_ranges.items():
            if min_value > 0:
                plt.axvline(min_value, color=self.structure_colors.get(struct_type, 'gray'), linestyle='--', alpha=0.7, label=f'{struct_type} threshold: {min_value}')
        if log_sacle:
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
        