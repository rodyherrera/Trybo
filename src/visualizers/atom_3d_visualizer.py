from typing import Tuple, Optional
from core.base_parser import BaseParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

class Atom3DVisualizer:
    def __init__(self, parser: BaseParser, property_column: Optional[int] = None):
        self.parser = parser
        self.property_column = property_column
        self.fig = None
        self.ax = None
    
    def visualize(self, output_file: Optional[str] = None, threshold: float = 8.0, 
                  show_planes: bool = True, view_angle: Optional[Tuple[float, float]] = None, 
                  colorbar_range: Optional[Tuple[float, float]] = None, 
                  property_name: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
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
                property_name = "Property Value"
        
        # Create figure
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Extract data
        x = data[:, 2]  # Assuming 3rd column is x
        y = data[:, 3]  # Assuming 4th column is y
        z = data[:, 4]  # Assuming 5th column is z
        values = data[:, self.property_column]
        
        # Set up colormap
        if colorbar_range is None:
            vmin = 0
            vmax = threshold * 2
        else:
            vmin, vmax = colorbar_range
        
        normalized_colors = colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cmx.ScalarMappable(norm=normalized_colors, cmap='viridis_r')
        
        # Plot atoms
        scatter = self.ax.scatter(x, y, z, c=values, s=20, cmap='viridis_r', 
                                 norm=normalized_colors, alpha=0.7)
        
        # Plot reference planes if requested
        if show_planes:
            self._add_reference_planes(x, y, z, box_size)
        
        # Labels and title
        self.ax.set_xlabel('X Position (Å)')
        self.ax.set_ylabel('Y Position (Å)')
        self.ax.set_zlabel('Z Position (Å)')
        self.ax.set_title(f'Atom Visualization: {property_name}')
        
        # Add colorbar
        colorbar = plt.colorbar(scalar_map, ax=self.ax, pad=0.1)
        colorbar.set_label(property_name)
        
        # Add threshold marker if appropriate
        if property_name and 'centro' in property_name.lower():
            colorbar.ax.axhline(y=threshold/vmax, color='red', linestyle='--', linewidth=2)
            colorbar.ax.text(0.5, threshold/vmax + 0.02, f'Threshold = {threshold}', 
                            transform=colorbar.ax.transAxes, ha='center', va='bottom', color='red')
        
        # Set specific view angle if provided
        if view_angle is not None:
            self.ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        plt.tight_layout()
        
        # Save figure if output file is specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f'Figure saved to {output_file}')
        
        return self.fig, self.ax
    
    def _add_reference_planes(self, x, y, z, box_size):
        xmin, xmax = box_size[0]
        ymin, ymax = box_size[1]
        zmin, zmax = box_size[2]
        
        # Find approximate positions of lower and upper planes based on z-coordinates
        z_values = np.sort(z)
        lower_z = np.percentile(z_values, 10)
        upper_z = np.percentile(z_values, 90)
        
        # Create planes at approximate positions
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
        lower_plane = np.ones_like(xx) * lower_z
        upper_plane = np.ones_like(xx) * upper_z
        
        # Plot transparent planes
        self.ax.plot_surface(xx, yy, lower_plane, alpha=0.1, color='blue')
        self.ax.plot_surface(xx, yy, upper_plane, alpha=0.1, color='red')