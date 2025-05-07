from parsers.energy_parser import EnergyParser
from parsers.cna_parser import CommonNeighborAnalysisParser
from parsers.coordination_parser import CoordinationParser
from parsers.vonmises_parser import VonmisesParser
from parsers.debris_parser import DebrisParser
from parsers.hotspot_parser import HotspotParser
from parsers.centro_symmetric_parser import CentroSymmetricParser
from parsers.velocity_squared_parser import VelocitySquaredParser

from visualizers.cna_visualizer import CommonNeighborAnalysisVisualizer
from visualizers.coordination_visualizer import CoordinationVisualizer
from visualizers.debris_visualizer import DebrisVisualizer
from visualizers.energy_visualizer import EnergyVisualizer
from visualizers.hotspot_visualizer import HotspotVisualizer
from visualizers.vonmises_visualizer import VonmisesVisualizer
from visualizers.centro_symmetric_visualizer import CentroSymmetricVisualizer
from visualizers.velocity_squared_visualizer import VelocitySquaredVisualizer

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
            "hotspot": run_hotspot_analysis, 
            "vonmises": run_vonmises_analysis,
            "centro_symmetric": run_centro_symmetric_analysis,
            "velocity_squared": run_velocity_squared_analysis,
            "energy": run_energy_analysis
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
                    print(e)
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
    
    if not os.path.exists(cna_file):
        logger.error(f"CNA file not found: {cna_file}")
        return
    
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
    
    if not os.path.exists(coord_file):
        logger.error(f"Coordination file not found: {coord_file}")
        return
    
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
    
    if not os.path.exists(debris_file):
        logger.error(f"Debris file not found: {debris_file}")
        return
    
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

def run_hotspot_analysis(dump_folder, timestep=-1):
    """Run Hotspot analysis visualization"""
    logger.info("Initializing Hotspot analysis")
    hotspot_file = os.path.join(dump_folder, "hotspots.dump")
    
    if not os.path.exists(hotspot_file):
        logger.error(f"Hotspot file not found: {hotspot_file}")
        return
    
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
    
    if not os.path.exists(vonmises_file):
        logger.error(f"Von Mises file not found: {vonmises_file}")
        return
    
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

def run_centro_symmetric_analysis(dump_folder, timestep=-1):
    """Run Centro-Symmetric analysis visualization"""
    logger.info("Initializing Centro-Symmetric analysis")
    cs_file = os.path.join(dump_folder, "center_symmetric.dump")
    
    if not os.path.exists(cs_file):
        logger.error(f"Centro-Symmetric file not found: {cs_file}")
        return
        
    parser = CentroSymmetricParser(cs_file)
    visualizer = CentroSymmetricVisualizer(parser)
    
    # Generate all Centro-Symmetric visualizations
    logger.info("Generating Centro-Symmetric parameter distribution")
    visualizer.plot_centro_symmetric_distribution(timestep)
    visualizer.plot_centro_symmetric_distribution(timestep, log_scale=True)
    
    # If there are multiple timesteps, show evolution
    timesteps = parser.get_timesteps()
    if len(timesteps) > 1:
        logger.info("Generating defect evolution plots")
        visualizer.plot_defect_evolution()
    
    logger.info("Generating 3D visualization of crystal structure")
    visualizer.plot_defect_3d(timestep)
    
    logger.info("Generating visualization of defect regions")
    visualizer.plot_defect_regions(timestep)
    
    logger.info("Generating centro-symmetric heat maps")
    visualizer.plot_centro_symmetric_heatmaps(timestep)
    
    logger.info("Comparing defects between groups")
    visualizer.plot_defect_by_groups(timestep)
    
    logger.info("Generating defect profile along Z-axis")
    visualizer.plot_defect_profile(timestep, axis='z')
    
    logger.info("Generating nanoparticle-specific analysis")
    visualizer.plot_centro_symmetric_distribution(timestep, group='nanoparticle')
    visualizer.plot_defect_regions(timestep, group='nanoparticle')
    
    if len(timesteps) > 1:
        logger.info("Generating nanoparticle defect evolution")
        visualizer.plot_defect_evolution(group='nanoparticle')

def run_velocity_squared_analysis(dump_folder, timestep=-1):
    """Run Velocity Squared analysis visualization"""
    logger.info("Initializing Velocity Squared analysis")
    vs_file = os.path.join(dump_folder, "velocity_squared.dump")
    
    if not os.path.exists(vs_file):
        logger.error(f"Velocity Squared file not found: {vs_file}")
        return
        
    parser = VelocitySquaredParser(vs_file)
    visualizer = VelocitySquaredVisualizer(parser)
    
    # Generate all Temperature visualizations
    logger.info("Generating temperature distribution plot")
    visualizer.plot_temperature_distribution(timestep)
    
    # If there are multiple timesteps, show evolution
    timesteps = parser.get_timesteps()
    if len(timesteps) > 1:
        logger.info("Generating temperature evolution plots")
        visualizer.plot_temperature_evolution()
    
    logger.info("Generating 3D visualization of temperature")
    visualizer.plot_temperature_3d(timestep)
    
    logger.info("Generating visualization of hot spots")
    visualizer.plot_hot_spots(timestep, threshold_percentile=95)
    
    logger.info("Generating temperature heat maps")
    visualizer.plot_temperature_heatmaps(timestep)
    
    if len(timesteps) > 1:
        logger.info("Generating temperature comparison between groups")
        visualizer.plot_temperature_by_groups()
    
    logger.info("Generating temperature gradient along Z-axis")
    visualizer.plot_temperature_gradient(timestep, axis='z')
    
    logger.info("Generating nanoparticle-specific analysis")
    visualizer.plot_temperature_distribution(timestep, group='nanoparticle')
    visualizer.plot_hot_spots(timestep, threshold_percentile=95, group='nanoparticle')

def run_energy_analysis(dump_folder, timestep=-1):
    """Run Energy analysis visualization"""
    logger.info("Initializing Energy analysis")
    energy_file = os.path.join(dump_folder, "energy.dump")
    
    if not os.path.exists(energy_file):
        logger.error(f"Energy file not found: {energy_file}")
        return
        
    parser = EnergyParser(energy_file)
    visualizer = EnergyVisualizer(parser)
    
    # Generate all Energy visualizations
    logger.info("Generating energy distribution plots")
    visualizer.plot_energy_distribution(timestep, energy_type='kinetic')
    visualizer.plot_energy_distribution(timestep, energy_type='potential')
    visualizer.plot_energy_distribution(timestep, energy_type='total')
    
    # If there are multiple timesteps, show evolution
    timesteps = parser.get_timesteps()
    if len(timesteps) > 1:
        logger.info("Generating energy evolution plots")
        visualizer.plot_energy_evolution(energy_type='kinetic')
        visualizer.plot_energy_evolution(energy_type='potential')
        visualizer.plot_energy_evolution(energy_type='total')
    
    logger.info("Generating 3D energy visualizations")
    visualizer.plot_energy_3d(timestep, energy_type='kinetic')
    visualizer.plot_energy_3d(timestep, energy_type='potential')
    visualizer.plot_energy_3d(timestep, energy_type='total')
    
    logger.info("Generating high energy region visualizations")
    visualizer.plot_high_energy_regions(timestep, energy_type='kinetic')
    visualizer.plot_high_energy_regions(timestep, energy_type='potential')
    visualizer.plot_high_energy_regions(timestep, energy_type='total')
    
    logger.info("Generating energy heat maps")
    visualizer.plot_energy_heatmaps(timestep, energy_type='total')
    
    if len(timesteps) > 1:
        logger.info("Generating energy comparison between groups")
        visualizer.plot_energy_by_groups(energy_type='total')
    
    logger.info("Generating energy profile along Z-axis")
    visualizer.plot_energy_profile(timestep, axis='z')
    
    logger.info("Generating energy type comparison")
    visualizer.plot_energy_comparison(timestep)
    
    logger.info("Generating nanoparticle-specific analysis")
    visualizer.plot_energy_distribution(timestep, group='nanoparticle')
    visualizer.plot_high_energy_regions(timestep, energy_type='total', group='nanoparticle')

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
        choices=['all', 'cna', 'coordination', 'debris', 'hotspot', 'vonmises', 
                'centro_symmetric', 'velocity_squared', 'energy'],
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