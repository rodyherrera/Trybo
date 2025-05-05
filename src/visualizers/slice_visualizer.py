from typing import Optional, Tuple, Dict, Any
from core.base_parser import BaseParser
import numpy as np
import matplotlib.pyplot as plt

class SliceVisualizer:
    def __init__(self, parser: BaseParser, property_column: Optional[int] = None):
        self.parser = parser
        self.property_column = property_column
        self.fig = None
        self.ax = None
    
    def visualize(self, slice_dim: str = 'z', slice_position: Optional[float] = None, 
                  slice_thickness: float = 2.0, output_file: Optional[str] = None,
                  threshold: float = 8.0, property_name: Optional[str] = None,
                  colorbar_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        # Get data
        data = self.parser.get_data()
        box_size = self.parser.get_box_size()
        
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
                property_name = 'Property Value'
        
        # Define which dimensions to plot based on slice_dim
        if slice_dim == 'x':
            dim_idx = 2
            plot_dims = [3, 4]
            xlabel, ylabel = 'Y Position (Å)', 'Z Position (Å)'
        elif slice_dim == 'y':
            dim_idx = 3
            plot_dims = [2, 4]
            xlabel, ylabel = 'X Position (Å)', 'Z Position (Å)'
        else:
            dim_idx = 4
            plot_dims = [2, 3]
            xlabel, ylabel = 'X Position (Å)', 'Y Position (Å)'
        
        # Determine slice position if not provided
        if slice_position is None:
            slice_position = np.mean(data[:, dim_idx])
        
        # Filter atoms within the slice
        slice_min = slice_position - slice_thickness/2
        slice_max = slice_position + slice_thickness/2
        slice_atoms = data[(data[:, dim_idx] >= slice_min) & 
                         (data[:, dim_idx] <= slice_max)]
        
        print(f'\nCreating {slice_dim}-slice at position {slice_position:.2f} with thickness {slice_thickness}')
        print(f'Number of atoms in slice: {len(slice_atoms)}')
        
        if len(slice_atoms) == 0:
            print('No atoms found in the slice!')
            return {'success': False, 'error': 'No atoms found in slice'}
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Set up colorbar range
        if colorbar_range is None:
            vmin = 0
            vmax = threshold * 2
        else:
            vmin, vmax = colorbar_range
        
        # Scatter plot colored by property value
        scatter = self.ax.scatter(slice_atoms[:, plot_dims[0]], slice_atoms[:, plot_dims[1]], 
                              c=slice_atoms[:, self.property_column], s=30, cmap='viridis_r', 
                              vmin=vmin, vmax=vmax, alpha=0.8)
        
        # Add colorbar
        colorbar = plt.colorbar(scatter, ax=self.ax)
        colorbar.set_label(property_name)
        
        # Add threshold marker on colorbar if appropriate
        if property_name and 'centro' in property_name.lower():
            colorbar.ax.axhline(y=threshold/(vmax - vmin), color='red', linestyle='--', linewidth=2)
        
        # Add labels and title
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(f'Slice at {slice_dim}={slice_position:.2f} ± {slice_thickness/2:.2f} Å')
        self.ax.grid(True, alpha=0.3)
        
        # Add legend for threshold if appropriate
        if property_name and 'centro' in property_name.lower():
            self.ax.plot([], [], color='red', linestyle='--', 
                      label=f'Threshold = {threshold}')
            self.ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save figure if output file is specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f'Slice figure saved to {output_file}')
        
        # Return slice information
        return {
            'success': True,
            'slice_dim': slice_dim,
            'slice_position': slice_position,
            'slice_thickness': slice_thickness,
            'num_atoms': len(slice_atoms),
            'output_file': output_file
        }
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None