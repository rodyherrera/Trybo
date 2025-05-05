import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import argparse

def read_dump_file(filename):
    print(f'Reading file: {filename}')
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Find the numbers of atoms
    total_atoms = int(lines[3])
    print(f'Total atoms: {total_atoms}')

    # Find the box dimensions
    box_size_x = [float(x) for x in lines[5].split()]
    box_size_y = [float(x) for x in lines[6].split()]
    box_size_z = [float(x) for x in lines[7].split()]

    # Find where the atoms data begins
    # Usually, the data starts at line 9 in LAMMPS dump files
    start_line = 9
    
    # Read atom data
    atom_data = []
    for i in range(start_line, start_line + total_atoms):
        if i >= len(lines):
            break

        values = lines[i].split()

        # At least id, type, x, y, z, c_centro_symmetric
        if len(values) >= 6:
            try:
                atom_id = int(values[0])
                atom_type = int(values[1])
                x = float(values[2])
                y = float(values[3])
                z = float(values[4])
                center = float(values[5])
                atom_data.append([ atom_id, atom_type, x, y, z, center ])
            except (ValueError, IndexError) as e:
                print(f'Error parsing line {i}: {lines[i]} - {e}')

    return np.array(atom_data), [box_size_x, box_size_y, box_size_z]

def visualize_atoms_3d(atom_data, box_size, output_file=None, threshold=8.0, show_planes=True, view_angle=None, colorbar_range=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract adata
    x = atom_data[:, 2]
    y = atom_data[:, 3]
    z = atom_data[:, 4]
    center = atom_data[:, 5]

    # Set up colormap for center-symmetry parameter
    if colorbar_range is None:
        vmin = 0
        vmax = threshold * 2
    else:
        vmin, vmax = colorbar_range
    
    normalized_colors = colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cmx.ScalarMappable(norm=normalized_colors, cmap='viridis_r')

    # Plot atoms as points colored by center-symmetry value
    scatter = ax.scatter(x, y, z, c=center, s=20, cmap='viridis_r', norm=normalized_colors, alpha=0.7)

    # Plot references planes for clarity if requested
    # TODO: CHECK THIS!
    if show_planes:
        xmin, xmax = box_size[0]
        ymin, ymax = box_size[1]
        zmin, zmax = box_size[2]

        # Find approximate positions of lower and upper planes based on z-coordinates
        z_values = np.sort(z)
        lower_z = np.percentile(z_values, 10)
        upper_z = np.percentile(z_values, 90)

        # Create planes at approximates positions
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
        lower_plane = np.ones_like(xx) * lower_z
        upper_plane = np.ones_like(xx) * upper_z

        # Plot transparent planes
        ax.plot_surface(xx, yy, lower_plane, alpha=0.1, color='blue')
        ax.plot_surface(xx, yy, upper_plane, alpha=0.1, color='red')

    ax.set_xlabel('X Position (Å)')
    ax.set_ylabel('Y Position (Å)')
    ax.set_zlabel('Z Position (Å)')
    ax.set_title('Atom Visualization with Center-Symmetry Parameter')
    
    colorbar = plt.colorbar(scalar_map, ax=ax, pad=0.1)
    colorbar.set_label('Center-Symmetry Parameter')

    # Horizontal line at the threshold value
    colorbar.ax.axhline(y=threshold/vmax, color='red', linestyle='--', linewidth=2)
    colorbar.ax.text(0.5, threshold/vmax + 0.02, f'Threshold = {threshold}', transform=colorbar.ax.transAxes, ha='center', va='bottom', color='red')

    # Set specific view angle if provided
    # TODO: CHECK THIS!
    if view_angle is not None:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    plt.tight_layout()

    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {output_file}')
    
    return fig, ax

def analyze_center_symmetry(atom_data, threshold=8.0):
    center = atom_data[:, 5]

    # Calculate statistics 
    mean = np.mean(center)
    median = np.median(center)
    max = np.max(center)
    min = np.min(center)
    std = np.std(center)

    # Count atoms with center-symmetry above threshold (defects)
    defect_count = np.sum(center > threshold)
    defect_percentage = (defect_count / len(center)) * 100

    print(f'Center-Symmetry Analaysis:')
    print(f'Mean: {mean:.4f}')
    print(f'Meadian: {median:.4f}')
    print(f'Max: {max:.4f}')
    print(f'Min: {min:.4f}')
    print(f'Standard Deviation: {std:.4f}')
    print(f'Atoms with Center-Symmetry > {threshold} (defects): {defect_count} ({defect_percentage:.2f})%')

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(center, bins=50, color='skyblue', alpha=0.7)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Centro-Symmetry Parameter')
    plt.ylabel('Number of Atoms')
    plt.title('Distribution of Centro-Symmetry Parameter')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return {
        'mean': mean,
        'median': median,
        'max': max,
        'min': min,
        'std': std,
        'defect_count': defect_count,
        'defect_percentage': defect_percentage
    }

def create_slice_view(atom_data, slice_dim='z', slice_position=None, slice_thickness=2.0, output_file=None, threshold=8.0):
    # Define which dimensions to plot based on slice_dim
    if slice_dim == 'x':
        dim_idx = 2
        plot_dims = [3, 4]
        xlabel = 'Y Position (Å)'
        ylabel = 'Z Position (Å)'
    elif slice_dim == 'y':
        dim_idx = 3
        plot_dims = [2, 4]
        xlabel = 'X Position (Å)'
        ylabel = 'Z Position (Å)'
    else:
        dim_idx = 4
        plot_dims = [2, 3]
        xlabel = 'X Position (Å)'
        ylabel = 'Y Position (Å)'

    # Determine slice position if not provided
    if slice_position is None:
        slice_position = np.mean(atom_data[:, dim_idx])

    # Filter atoms within the slice
    slice_min = slice_position - slice_thickness / 2
    slice_max = slice_position + slice_thickness / 2
    slice_atoms = atom_data[(atom_data[:, dim_idx] >= slice_min) & (atom_data[:, dim_idx] <= slice_max)]
    print(f'Creating {slice_dim}-slice at position {slice_position} with thickness {slice_thickness}')
    print(f'Number of atoms in slice: {len(slice_atoms)}')

    if len(slice_atoms) == 0:
        print('No atoms found in the slice!')
        return None
    
    plt.figure(figsize=(10, 8))

    # Colored by center-symmetry parameter
    scatter = plt.scatter(slice_atoms[:, plot_dims[0]], slice_atoms[:, plot_dims[1]], 
                c=slice_atoms[:, 5], s=30, cmap='viridis_r', vmin=0, vmax=threshold * 2, alpha=0.8)

    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Center-Symmetry Parameter')

    # Threshold marker on colorbar
    colorbar.ax.axhline(y=threshold/(threshold * 2), color='red', linestyle='--', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Slice at {slice_dim}={slice_position:.2f} ± {slice_thickness/2:.2f} Å')
    plt.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'Slice figure saved to {output_file}')
    
    return slice_atoms

def main():
    parser = argparse.ArgumentParser(description='Visualize and analyze Trybo dump files')
    parser.add_argument('dump_file', help='LAMMPS dump file or pattern (use wildcards for multiple files)')
    parser.add_argument('--output', '-o', default='centro_symmetry_analysis', help='Output directory or file prefix')
    parser.add_argument('--threshold', '-t', type=float, default=8.0, help='Centro-symmetry threshold for defect identification')
    parser.add_argument('--slice-pos', type=float, help='Position for slice view')
    parser.add_argument('--slice-thickness', type=float, default=2.0, help='Thickness for slice view')
    
    args = parser.parse_args()

    atom_data, box_size = read_dump_file(args.dump_file)
    stats = analyze_center_symmetry(atom_data, args.threshold)

    plt.savefig(f'{args.output}_histogram.png', dpi=300, bbox_inches='tight')
    visualize_atoms_3d(atom_data, box_size, f'{args.output}_3d.png', args.threshold)
    create_slice_view(
        atom_data=atom_data,
        slice_dim=args.slice_dim,
        slice_position=args.slice_pos,
        slice_thickness=args.slice_thickness,
        output_file=f'{args.output}_slice_{args.slice_dim}.png',
        threshold=args.threshold
    )

    plt.show()

if __name__ == '__main__':
    main()