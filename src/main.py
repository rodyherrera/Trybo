from parsers.energy_parser import EnergyParser
from visualizers.energy_visualizer import EnergyVisualizer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Visualize and analyze centro-symmetric dump files')
    parser.add_argument('dump_file', help='LAMMPS dump file path')
    
    args = parser.parse_args()
    
    parser = EnergyParser(args.dump_file)
    visualizer = EnergyVisualizer(parser)
    visualizer.plot_kinetic_energy()
    
if __name__ == '__main__':
    main()