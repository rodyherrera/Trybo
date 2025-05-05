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
    
    parser = EnergyParser(args.dump_file)
    energy_visualizer = EnergyVisualizer(parser)
    energy_visualizer.plot_energy_evolution_statistics()
    
if __name__ == '__main__':
    main()