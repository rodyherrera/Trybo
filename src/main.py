from parsers.energy_parser import EnergyParser
from parsers.cna_parser import CommonNeighborAnalysisParser
from parsers.coordination_parser import CoordinationParser
from parsers.vonmises_parser import VonmisesParser
from parsers.debris_parser import DebrisParser
from parsers.hotspot_parser import HotspotParser

from visualizers.cna_visualizer import CommonNeighborAnalysisVisualizer
from visualizers.coordination_visualizer import CoordinationVisualizer
from visualizers.debris_visualizer import DebrisVisualizer
from visualizers.energy_visualizer import EnergyVisualizer
from visualizers.hotspot_visualizer import HotspotVisualizer
from visualizers.vonmises_visualizer import VonmisesVisualizer

import argparse
import os
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("nanoparticle_wear")

def run_analysis(dump_folder, analysis_type=None, timestep=-1):
    """
    Run the specified analysis on data from the dump folder.
    
    Args:
        dump_folder (str): Path to the dump directory containing analysis files
        analysis_type (str, optional): Type of analysis to run ('all' or specific analysis name)
        timestep (int, optional): Timestep to analyze, -1 for last timestep
    """
    start_time = time.time()
    output_dir = os.path.join(dump_folder, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Change to output directory for saving visualizations
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Dictionary mapping analysis types to functions
        analysis_functions = {
            "cna": run_cna_analysis,
            "coordination": run_coordination_analysis,
            "debris": run_debris_analysis,
            "energy": run_energy_analysis,
            "hotspot": run_hotspot_analysis, 
            "vonmises": run_vonmises_analysis
        }
        
        if analysis_type is None or analysis_type == "all":
            # Run all analyses
            logger.info(f"Running all analyses on data from {dump_folder}")
            for name, func in analysis_functions.items():
                try:
                    logger.info(f"Starting {name} analysis")
                    func(dump_folder, timestep)
                    logger.info(f"Completed {name} analysis")
                except Exception as e:
                    logger.error(f"Error in {name} analysis: {e}")
        elif analysis_type in analysis_functions:
            # Run specific analysis
            logger.info(f"Running {analysis_type} analysis on data from {dump_folder}")
            analysis_functions[analysis_type](dump_folder, timestep)
        else:
            logger.error(f"Unknown analysis type: {analysis_type}")
            print(f"Available analysis types: {', '.join(analysis_functions.keys())}")
            return
            
        elapsed_time = time.time() - start_time
        logger.info(f"All analyses completed in {elapsed_time:.2f} seconds")
        print(f"Analysis results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
    finally:
        os.chdir(original_dir)

def run_cna_analysis(dump_folder, timestep=-1):
    """Run Common Neighbor Analysis visualization"""
    logger.info("Initializing CNA analysis")
    cna_file = os.path.join(dump_folder, "cna.dump")
    parser = CommonNeighborAnalysisParser(cna_file)
    visualizer = CommonNeighborAnalysisVisualizer(parser)
    
    # Generate all CNA visualizations
    logger.info("Generating CNA distribution plot")
    visualizer.plot_structure_distribution(timestep)
    
    logger.info("Generating CNA evolution plot")
    visualizer.plot_structure_evolution()
    
    logger.info("Generating CNA spatial distribution heatmap")
    visualizer.plot_structure_heatmap(timestep)
    
    logger.info("Generating CNA structure comparison")
    visualizer.plot_structure_comparison(0, timestep)

def run_coordination_analysis(dump_folder, timestep=-1):
    """Run Coordination analysis visualization"""
    logger.info("Initializing Coordination analysis")
    coord_file = os.path.join(dump_folder, "coordination.dump")
    parser = CoordinationParser(coord_file)
    visualizer = CoordinationVisualizer(parser)
    
    # Generate all Coordination visualizations
    logger.info("Generating coordination distribution plot")
    visualizer.plot_coord_distribution(timestep)
    
    logger.info("Generating coordination evolution plot")
    visualizer.plot_coord_evolution()
    
    logger.info("Generating coordination spatial distribution")
    visualizer.plot_coord_spatial(timestep)
    
    logger.info("Generating atom classification by coordination")
    visualizer.plot_atom_classification(timestep)
    
    logger.info("Generating coordination range distribution")
    visualizer.plot_coord_ranges(timestep)
    
    logger.info("Generating coordination comparison")
    visualizer.plot_coord_comparison(0, timestep)

def run_debris_analysis(dump_folder, timestep=-1):
    """Run Debris Cluster analysis visualization"""
    logger.info("Initializing Debris analysis")
    debris_file = os.path.join(dump_folder, "debris_clusters.dump")
    parser = DebrisParser(debris_file)
    visualizer = DebrisVisualizer(parser)
    
    # Generate all Debris visualizations
    logger.info("Generating cluster evolution plot")
    visualizer.plot_cluster_evolution()
    
    logger.info("Generating cluster size distribution")
    visualizer.plot_cluster_size_distribution(timestep, min_size=2)
    
    logger.info("Generating 3D cluster visualization")
    visualizer.plot_3d_cluster_visualization(timestep, min_size=3)
    
    logger.info("Generating 2D projections of clusters")
    visualizer.plot_2d_projections(timestep, min_size=3)
    
    logger.info("Generating largest clusters information table")
    visualizer.plot_largest_clusters_info(timestep)

def run_energy_analysis(dump_folder, timestep=-1):
    """Run Energy analysis visualization"""
    logger.info("Initializing Energy analysis")
    energy_file = os.path.join(dump_folder, "energy.dump")
    parser = EnergyParser(energy_file)
    visualizer = EnergyVisualizer(parser)
    
    # Generate all Energy visualizations
    logger.info("Generating kinetic energy plot")
    visualizer.plot_kinetic_energy()
    
    logger.info("Generating potential energy plot")
    visualizer.plot_potential_energy()
    
    logger.info("Generating total energy plot")
    visualizer.plot_total_energy()

def run_hotspot_analysis(dump_folder, timestep=-1):
    """Run Hotspot analysis visualization"""
    logger.info("Initializing Hotspot analysis")
    hotspot_file = os.path.join(dump_folder, "hotspots.dump")
    parser = HotspotParser(hotspot_file)
    visualizer = HotspotVisualizer(parser)
    
    # Generate all Hotspot visualizations
    logger.info("Generating energy distribution plot")
    visualizer.plot_energy_distribution(timestep)
    
    logger.info("Generating hotspot evolution plot")
    visualizer.plot_hotspot_evolution()
    
    logger.info("Generating hotspot spatial distribution")
    visualizer.plot_hotspot_spatial(timestep)
    
    logger.info("Generating 3D hotspot clusters")
    visualizer.plot_hotspot_clusters_3d(timestep)
    
    logger.info("Generating hotspot heatmap")
    visualizer.plot_hotspot_heatmap(timestep)

def run_vonmises_analysis(dump_folder, timestep=-1):
    """Run von Mises stress analysis visualization"""
    logger.info("Initializing von Mises analysis")
    vonmises_file = os.path.join(dump_folder, "vonmises.dump")
    parser = VonmisesParser(vonmises_file)
    visualizer = VonmisesVisualizer(parser)
    
    # Generate all von Mises visualizations
    logger.info("Generating stress evolution plot")
    visualizer.plot_stress_evolution()
    
    logger.info("Generating stress heatmaps")
    visualizer.plot_stress_heatmaps(timestep)
    
    logger.info("Generating stress distribution")
    visualizer.plot_stress_distribution(timestep)
    
    logger.info("Generating stress by groups")
    visualizer.plot_stress_by_groups()
    
    logger.info("Generating 3D stress visualization")
    visualizer.plot_stress_3d(timestep)
    visualizer.plot_stress_3d(timestep, group='nanoparticle', percentile_threshold=90)
    
    logger.info("Generating stress by layer")
    visualizer.plot_stress_by_layer(timestep, axis='z')

def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize nanoparticle wear simulation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py /path/to/dumps/2025-05-05_05-36-35
  python main.py /path/to/dumps/2025-05-05_05-36-35 --analysis cna
  python main.py /path/to/dumps/2025-05-05_05-36-35 --timestep -1
        """
    )
    
    parser.add_argument(
        'dump_folder', 
        help='The directory containing dump files generated by the LAMMPS simulation'
    )
    
    parser.add_argument(
        '--analysis', '-a',
        choices=['all', 'cna', 'coordination', 'debris', 'energy', 'hotspot', 'vonmises'],
        default='all',
        help='Type of analysis to run (default: all)'
    )
    
    parser.add_argument(
        '--timestep', '-t',
        type=int,
        default=-1,
        help='Timestep to analyze (default: -1, the last timestep)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate dump folder exists
    dump_path = Path(args.dump_folder)
    if not dump_path.exists() or not dump_path.is_dir():
        logger.error(f"Error: Dump folder '{args.dump_folder}' does not exist or is not a directory")
        return 1
    
    # Run selected analysis
    run_analysis(args.dump_folder, args.analysis, args.timestep)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())