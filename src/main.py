from parsers.center_symmetry_parser import CenterSymmetryParser
from analyzers.statistical_analyzer import StatisticalAnalyzer
from visualizers.atom_3d_visualizer import Atom3DVisualizer
from visualizers.slice_visualizer import SliceVisualizer
from visualizers.histogram_visualizer import HistogramVisualizer
from parsers.energy_parser import EnergyParser
from visualizers.energy_visualizer import EnergyVisualizer
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Visualize and analyze centro-symmetric dump files')
    parser.add_argument('dump_file', help='LAMMPS dump file path')
    parser.add_argument('--output', '-o', default=None, help='Output file prefix')
    parser.add_argument('--threshold', '-t', type=float, default=8.0, help='Threshold value')
    parser.add_argument('--slice-dim', choices=['x', 'y', 'z'], default='z', help='Dimension for slice view')
    parser.add_argument('--slice-pos', type=float, help='Position for slice view')
    parser.add_argument('--slice-thickness', type=float, default=2.0, help='Thickness for slice view')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.splitext(os.path.basename(args.dump_file))[0] + '_analysis'
    
    parser = CenterSymmetryParser(args.dump_file, threshold=args.threshold)
    
    analyzer = StatisticalAnalyzer(parser)
    stats = analyzer.analyze(threshold=args.threshold)
    
    print('\nCentro-Symmetry Analysis:')
    print(f'Mean: {stats['mean']:.4f}')
    print(f'Median: {stats['median']:.4f}')
    print(f'Max: {stats['max']:.4f}')
    print(f'Min: {stats['min']:.4f}')
    print(f'Standard Deviation: {stats['std']:.4f}')
    print(f'Atoms with Centro-Symmetry > {args.threshold} (defects): '
          f'{stats['high_count']} ({stats['high_percentage']:.2f}%)')
    
    histogram_visualizer = HistogramVisualizer(parser)
    histogram_visualizer.visualize(
        output_file=f'{args.output}_histogram.png',
        threshold=args.threshold,
        property_name='Centro-Symmetry Parameter'
    )
    
    atom_visualizer = Atom3DVisualizer(parser)
    atom_visualizer.visualize(
        output_file=f'{args.output}_3d.png',
        threshold=args.threshold,
        property_name='Centro-Symmetry Parameter'
    )
    
    slice_visualizer = SliceVisualizer(parser)
    slice_visualizer.visualize(
        slice_dim=args.slice_dim,
        slice_position=args.slice_pos,
        slice_thickness=args.slice_thickness,
        output_file=f'{args.output}_slice_{args.slice_dim}.png',
        threshold=args.threshold,
        property_name='Centro-Symmetry Parameter'
    )
    
if __name__ == '__main__':
    main()