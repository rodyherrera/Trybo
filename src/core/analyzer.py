from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import os
import sys
import time
import yaml
import visualizers
import logging
import parsers

class Analyzer:
    def __init__(self, config_path: str = None, dump_folder: str = None):
        self.logger = logging.getLogger('Analyzer')
        self._setup_logging()

        # Initialize configuration
        self.config = {}
        if config_path:
            self.load_config(config_path)
        
        # Override dump folder if provided
        if dump_folder:
            self.set_dump_folder(dump_folder)
        
        self.analysis_registry = {
            'cna': self.run_cna_analysis,
            'coordination': self.run_coordination_analysis,
            'debris': self.run_debris_analysis,
            'hotspot': self.run_hotspot_analysis,
            'vonmises': self.run_vonmises_analysis,
            'centro_symmetric': self.run_centro_symmetric_analysis,
            'velocity_squared': self.run_velocity_squared_analysis,
            'energy': self.run_energy_analysis
        }
    
    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def load_config(self, config_path: str) -> bool:
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.logger.info(f'Configuration loaded from {config_path}')
                return True
        except Exception as e:
            self.logger.error(f'Error loading configuration from {config_path}: {str(e)}')
            return False
    
    def set_dump_folder(self, dump_folder: str):
        if not os.path.exists(dump_folder) or not os.path.isdir(dump_folder):
            set.logger.error(f'Dump folder does not exist or is not a directory: {dump_folder}')
            return False
        if 'analysis' not in self.config:
            self.config['analysis'] = {}
        self.config['analysis']['dump_folder'] = dump_folder
        self.logger.info(f'Dump folder set to: {dump_folder}')
        return True
    
    def set_output_folder(self, output_folder: str = None) -> str:
        if not output_folder:
            # Use dump folder with "analysis_results" subfolder if not specified
            dump_folder = self.config.get('analysis', {}).get('dump_folder')
            if not dump_folder:
                self.logger.error('No dump folder specified in configuration')
                return None
            output_folder = os.path.join(dump_folder, 'anaysis_results')
        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        if 'analysis' not in self.config:
            self.config['analysis'] = {}
        self.config['analysis']['output_folder'] = output_folder
        self.logger.info(f'Output folder set to: {output_folder}')
        return output_folder
    
    def run_analysis(self, analysis_type: str = None, timestep: int = -1) -> bool:
        start_time = time.time()
        
        # Get dump folder from config
        dump_folder = self.config.get('analysis', {}).get('dump_folder')
        if not dump_folder:
            self.logger.error("No dump folder specified in configuration")
            return False
        
        # Set output folder if not already set
        output_folder = self.config.get('analysis', {}).get('output_folder')
        if not output_folder:
            output_folder = self.set_output_folder()
            if not output_folder:
                return False
        
        # Change to output directory for saving visualizations
        original_dir = os.getcwd()
        os.chdir(output_folder)
        
        success = True
        try:
            if analysis_type is None or analysis_type == 'all':
                # Run all analyses
                self.logger.info(f'Running all analyses on data from {dump_folder}')
                for name, function in self.analysis_registry.items():
                    self.logger.info(f'Start "{name}" analysis')
                    try:
                        function(dump_folder, timestep)
                        self.logger.info(f'Completed "{name}" analysis')
                    except Exception as e:
                        self.logger.error(f'Error in {name} analysis: {str(e)}')
                        success = False
            elif analysis_type in self.analysis_registry:
                # Run sepecific analysis
                self.logger.info(f'Running {analysis_type} analysis on data from {dump_folder}')
                try:
                    self.analysis_registry[analysis_type](dump_folder, timestep)
                except Exception as e:
                    self.logger.error(f'Error in {analysis_type} analysis: {str(e)}')
                    success = False
            else:
                self.logger.error(f'Unknown analysis type: {analysis_type}')
                self.logger.info(f'Available analysis types: {", ".join(self.analysis_registry.keys())}')
                success = False
        finally:
            # Return to original directory
            os.chdir(original_dir)
        elapsed_time = time.time() - start_time
        self.logger.info(f'Analysis completed in {elapsed_time:.2f} seconds')
        self.logger.info(f'Analysis results saved to {output_folder}')
        return success
    
    def run_cna_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing CNA analysis')
        cna_file = os.path.join(dump_folder, 'cna.dump')

        if not os.path.exists(cna_file):
            self.logger.error(f'CNA file not found: {cna_file}')
            return False
        
        parser = parsers.CommonNeighborAnalysisParser(cna_file)
        visualizer = visualizers.CommonNeighborAnalysisVisualizer(parser)

        self.logger.info('Generating CNA distribution plot')
        visualizer.plot_structure_distribution(timestep)

        self.logger.info('Generating CNA evolution plot')
        visualizer.plot_structure_evolution()

        self.logger.info('Generating CNA spatial distribution heatmap')
        visualizer.plot_structure_heatmap(timestep)

        self.logger.info('Generating CNA structure comparison')
        visualizer.plot_structure_comparison(0, timestep)

        return True

    def run_coordination_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing Coordination analysis')
        coordination_file = os.path.join(dump_folder, 'coordination.dump')

        if not os.path.exists(coord_file):
            self.logger.error(f'Coordination file not found: {coord_file}')
            return False
        
        parser = parsers.CoordinationParser(coord_file)
        visualizer = parsers.CoordinationVisualizer(parser)

        self.logger.info('Generating coordination distribution plot')
        visualizer.plot_coord_distribution(timestep)

        self.logger.info('Generating coordination evolution plot')
        visualizer.plot_coord_evolution()

        self.logger.info('Generating coordination spatial distribution')
        visualizer.plot_coord_spatial(timestep)

        self.logger.info('Generating atom classification by coordination')
        visualizer.plot_atom_classification(timestep)

        self.logger.info('Generating coordiantion range distribution')
        visualizer.plot_coord_ranges(timestep)

        self.logger.info('Generating coordination comparison')
        visualizer.plot_coord_comparison(0, timestep)

        return True
    
    def run_debris_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing Debris analysis')
        debris_file = os.path.join(dump_folder, 'debris_clusters.dump')

        if not os.path.exists(debris_file):
            self.logger.error(f'Debris file not found: {debris_file}')
            return False
        
        parser = parsers.DebrisParser(debris_file)
        visualizer = visualizers.DebrisVisualizer(parser)

        self.logger.info('Generating cluster evolution plot')
        visualizer.plot_cluster_evolution()
        
        self.logger.info('Generating cluster size distribution')
        visualizer.plot_cluster_size_distribution(timestep, min_size=2)
        
        self.logger.info('Generating 3D cluster visualization')
        visualizer.plot_3d_cluster_visualization(timestep, min_size=3)
        
        self.logger.info('Generating 2D projections of clusters')
        visualizer.plot_2d_projections(timestep, min_size=3)
        
        self.logger.info('Generating largest clusters information table')
        visualizer.plot_largest_clusters_info(timestep)
        
        return True

    def run_hotspot_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing Hotspot analysis')
        hotspot_file = os.path.join(dump_folder, 'hotspot.dump')

        if not os.path.exists(hotspot_file):
            self.logger.error(f'Hotspot file not found: {hotspot_file}')
            return False

        parser = parsers.HotspotParser(hotspot_file)
        visualizer = visualizers.HotspotVisualizer(parser)

        self.logger.info('Generating energy distribution plot')
        visualizer.plot_energy_distribution(timestep)
        
        self.logger.info('Generating hotspot evolution plot')
        visualizer.plot_hotspot_evolution()
        
        self.logger.info('Generating hotspot spatial distribution')
        visualizer.plot_hotspot_spatial(timestep)
        
        self.logger.info('Generating 3D hotspot clusters')
        visualizer.plot_hotspot_clusters_3d(timestep)
        
        self.logger.info('Generating hotspot heatmap')
        visualizer.plot_hotspot_heatmap(timestep)
        
        return True

    def run_vonmises_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing von Mises analysis')
        vonmises_file = os.path.join(dump_folder, 'vonmises.dump')

        if not os.path.exists(vonmises_file):
            self.logger.error(f'Von Mises file not found: {vonmises_file}')
            return False
        
        parser = parsers.VonmisesParser(vonmises_file)
        visualizer = visualizers.VonmisesVisualizer(parser)

        self.logger.info('Generating stress evolution plot')
        visualizer.plot_stress_evolution()
        
        self.logger.info('Generating stress heatmaps')
        visualizer.plot_stress_heatmaps(timestep)
        
        self.logger.info('Generating stress distribution')
        visualizer.plot_stress_distribution(timestep)
        
        self.logger.info('Generating stress by groups')
        visualizer.plot_stress_by_groups()
        
        self.logger.info('Generating 3D stress visualization')
        visualizer.plot_stress_3d(timestep)
        visualizer.plot_stress_3d(timestep, group='nanoparticle', percentile_threshold=90)
        
        self.logger.info('Generating stress by layer')
        visualizer.plot_stress_by_layer(timestep, axis='z')

        return True

    def run_centro_symmetric_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing Centro-Symmetric analysis')
        centro_symmetric_file = os.path.join(dump_folder, 'center_symmetric.dump')

        if not os.path.exists(centro_symmetric_file):
            self.logger.error(f'Centro-Symmetric file not found: {centro_symmetric_file}')
            return False
        
        parser = parsers.CentroSymmetricParser(centro_symmetric_file)
        visualizer = visualizers.CentroSymmetricVisualizer(parser)

        self.logger.info('Generating Centro-Symmetric parameter distribution')
        visualizer.plot_centro_symmetric_distribution(timestep)
        visualizer.plot_centro_symmetric_distribution(timestep, log_scale=True)

        timesteps = parser.get_timesteps()
        if len(timesteps) > 1:
            self.logger.info('Generating defect evolution plots')
            visualizer.plot_defect_evolution()
            
            self.logger.info('Generating nanoparticle defect evolution')
            visualizer.plot_defect_evolution(group='nanoparticle')

        self.logger.info('Generating 3D visualization of crystal structure')
        visualizer.plot_defect_3d(timestep)

        self.logger.info('Generating visualization of defect regions')
        visualizer.plot_defect_regions(timestep)

        self.logger.info('Generating centro-symmetric heat maps')
        visualizer.plot_centro_symmetric_heatmaps(timestep)
        
        self.logger.info('Comparing defects between groups')
        visualizer.plot_defect_by_groups(timestep)
        
        self.logger.info('Generating defect profile along Z-axis')
        visualizer.plot_defect_profile(timestep, axis='z')
        
        self.logger.info('Generating nanoparticle-specific analysis')
        visualizer.plot_centro_symmetric_distribution(timestep, group='nanoparticle')
        visualizer.plot_defect_regions(timestep, group='nanoparticle')
    
        return True

    def run_velocity_squared_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing Velocity Squared analysis')
        velocity_squared_file = os.path.join(dump_folder, 'velocity_squared.dump')

        if not os.path.exists(velocity_squared_file):
            self.logger.error(f'Velocity Squared file not found: {velocity_squared_file}')
            return False
        
        parser = parsers.VelocitySquaredParser(velocity_squared_file)
        visualizer = visualizers.VelocitySquaredVisualizer(parser)

        self.logger.info('Generating temperature distribution plot')
        visualizer.plot_temperature_distribution(timestep)
        
        timesteps = parser.get_timesteps()
        if len(timesteps) > 1:
            self.logger.info('Generating temperature evolution plots')
            visualizer.plot_temperature_evolution()
        
        self.logger.info('Generating 3D visualization of temperature')
        visualizer.plot_temperature_3d(timestep)
        
        self.logger.info('Generating visualization of hot spots')
        visualizer.plot_hot_spots(timestep, threshold_percentile=95)
        
        self.logger.info('Generating temperature heat maps')
        visualizer.plot_temperature_heatmaps(timestep)
        
        if len(timesteps) > 1:
            self.logger.info('Generating temperature comparison between groups')
            visualizer.plot_temperature_by_groups()
        
        self.logger.info('Generating temperature gradient along Z-axis')
        visualizer.plot_temperature_gradient(timestep, axis='z')
        
        self.logger.info('Generating nanoparticle-specific analysis')
        visualizer.plot_temperature_distribution(timestep, group='nanoparticle')
        visualizer.plot_hot_spots(timestep, threshold_percentile=95, group='nanoparticle')
        
        return True

    def run_energy_analysis(self, dump_folder: str, timestep: int = -1) -> bool:
        self.logger.info('Initializing Energy analysis')
        energy_file = os.path.join(dump_folder, 'energy.dump')
        
        if not os.path.exists(energy_file):
            self.logger.error(f'Energy file not found: {energy_file}')
            return False
            
        parser = parsers.EnergyParser(energy_file)
        visualizer = visualizers.EnergyVisualizer(parser)
        
        self.logger.info('Generating energy distribution plots')
        visualizer.plot_energy_distribution(timestep, energy_type='kinetic')
        visualizer.plot_energy_distribution(timestep, energy_type='potential')
        visualizer.plot_energy_distribution(timestep, energy_type='total')
        
        timesteps = parser.get_timesteps()
        if len(timesteps) > 1:
            self.logger.info('Generating energy evolution plots')
            visualizer.plot_energy_evolution(energy_type='kinetic')
            visualizer.plot_energy_evolution(energy_type='potential')
            visualizer.plot_energy_evolution(energy_type='total')

            self.logger.info('Generating energy comparison between groups')
            visualizer.plot_energy_by_groups(energy_type='total')
    

        self.logger.info('Generating 3D energy visualizations')
        visualizer.plot_energy_3d(timestep, energy_type='kinetic')
        visualizer.plot_energy_3d(timestep, energy_type='potential')
        visualizer.plot_energy_3d(timestep, energy_type='total')
        
        self.logger.info('Generating high energy region visualizations')
        visualizer.plot_high_energy_regions(timestep, energy_type='kinetic')
        visualizer.plot_high_energy_regions(timestep, energy_type='potential')
        visualizer.plot_high_energy_regions(timestep, energy_type='total')
        
        self.logger.info('Generating energy heat maps')
        visualizer.plot_energy_heatmaps(timestep, energy_type='total')

        self.logger.info('Generating energy profile along Z-axis')
        visualizer.plot_energy_profile(timestep, axis='z')
        
        self.logger.info('Generating energy type comparison')
        visualizer.plot_energy_comparison(timestep)
        
        self.logger.info('Generating nanoparticle-specific analysis')
        visualizer.plot_energy_distribution(timestep, group='nanoparticle')
        visualizer.plot_high_energy_regions(timestep, energy_type='total', group='nanoparticle')

        return True