from typing import Optional, Tuple, Dict, Any, List, Union
from core.base_parser import BaseParser
import numpy as np
import matplotlib.pyplot as plt

class HistogramVisualizer:
    def __init__(self, parser: BaseParser, property_column: Optional[int] = None):
        self.parser = parser
        self.property_column = property_column
        self.fig = None
        self.ax = None
    
    def visualize(self, output_file: Optional[str] = None, threshold: Optional[float] = None,
                  property_name: Optional[str] = None, bins: int = 50, 
                  color: str = 'skyblue', additional_thresholds: Optional[List[Tuple[float, str, str]]] = None,
                  log_scale: bool = False) -> Dict[str, Any]:
        # Get data
        data = self.parser.get_data()
        
        # Determine property column if not specified
        if self.property_column is None:
            # Try to get property column from specialized parser
            if hasattr(self.parser, 'get_property_column_index'):
                self.property_column = self.parser.get_property_column_index()
            else:
                # Default to last column (common in LAMMPS dumps)
                self.property_column = data.shape[1] - 1
        
        # Get property name if not specified
        if property_name is None:
            if hasattr(self.parser, 'property_name'):
                property_name = self.parser.property_name
            else:
                property_name = "Property Value"
        
        # Extract property values
        values = data[:, self.property_column]
        
        # Calculate statistics
        stats = {
            'mean': np.mean(values),
            'median': np.median(values),
            'max': np.max(values),
            'min': np.min(values),
            'std': np.std(values)
        }
        
        # Add threshold-based statistics if threshold provided
        if threshold is not None:
            high_count = np.sum(values > threshold)
            high_percentage = (high_count / len(values)) * 100
            stats.update({
                'threshold': threshold,
                'high_count': high_count,
                'high_percentage': high_percentage
            })
        
        # Create histogram figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Get histogram counts for custom annotations
        hist_counts, bin_edges, _ = self.ax.hist(
            values, bins=bins, color=color, alpha=0.7,
            log=log_scale, edgecolor='black', linewidth=0.5
        )
        
        # Add main threshold line if specified
        if threshold is not None:
            self.ax.axvline(x=threshold, color='red', linestyle='--',
                         label=f'Threshold ({threshold})')
            
            # Find the bin containing the threshold
            bin_index = np.searchsorted(bin_edges, threshold) - 1
            if 0 <= bin_index < len(hist_counts):
                # Add annotation showing count and percentage
                y_pos = hist_counts[bin_index] * 1.1
                if log_scale and y_pos <= 0:
                    y_pos = 1 
                
                self.ax.annotate(
                    f"{high_percentage:.1f}% > {threshold}",
                    xy=(threshold, y_pos),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='red'),
                    fontweight='bold', color='red'
                )
                
        # Add additional threshold lines if specified
        if additional_thresholds:
            for value, color, label in additional_thresholds:
                self.ax.axvline(x=value, color=color, linestyle='--', label=label)
        
        # Add mean and median lines
        self.ax.axvline(x=stats['mean'], color='green', linestyle='-', linewidth=1.5,
                     label=f'Mean ({stats["mean"]:.2f})')
        self.ax.axvline(x=stats['median'], color='blue', linestyle=':', linewidth=1.5,
                     label=f'Median ({stats["median"]:.2f})')
        
        # Add labels and title
        self.ax.set_xlabel(property_name)
        self.ax.set_ylabel('Number of Atoms' + (' (log scale)' if log_scale else ''))
        self.ax.set_title(f'Distribution of {property_name}')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Add statistics text box
        stats_text = (
            f"Total atoms: {len(values)}\n"
            f"Mean: {stats['mean']:.4f}\n"
            f"Median: {stats['median']:.4f}\n"
            f"Std Dev: {stats['std']:.4f}\n"
            f"Min: {stats['min']:.4f}\n"
            f"Max: {stats['max']:.4f}"
        )
        if threshold is not None:
            stats_text += f"\nAbove threshold: {stats['high_count']} ({stats['high_percentage']:.2f}%)"
            
        self.ax.text(
            0.02, 0.98, stats_text, transform=self.ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save figure if output file is specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Histogram saved to {output_file}")
        
        return stats
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None