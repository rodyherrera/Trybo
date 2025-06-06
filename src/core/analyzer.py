from concurrent.futures import ProcessPoolExecutor, as_completed
from core.base_parser import BaseParser

import os
import time
import visualizers
import logging
import sys
import gc

class Analyzer:
    def __init__(self, yaml_config=None, dump_folder=None):
        self.logger = logging.getLogger('Analyzer')
        self._setup_logging()
        self.dump_folder = dump_folder

        self.parser = self._create_parser()
        self.yaml_config = yaml_config
        self.config = yaml_config.config if yaml_config else {}
        if yaml_config:
            self.logger.info('Using configuration from YamlConfig instance')

        if dump_folder:
            self.set_dump_folder(dump_folder)
        
        self.analysis_registry = self._register_analysis_methods()
        self.memory_efficient = True
        self.parallel_execution = False
        self.max_workers = os.cpu_count() or 1

    def _create_parser(self):
        analysis_file_path = os.path.join(self.dump_folder, 'analysis.lammpstrj')
        if not os.path.isfile(analysis_file_path):
            self.logger.error(f'Error: The file "analysis.lammpstrj" does not exist in the directory "{self.dump_folder}".')
            sys.exit(1)
        parser = BaseParser(analysis_file_path)
        return parser

    def enable_memory_efficient_mode(self, enabled=True):
        self.memory_efficient = enabled
        self.logger.info(f'Memory efficient mode: {"enabled" if enabled else "disabled"}')

    def enable_parallel_execution(self, enabled=True):
        self.parallel_execution = enabled
        self.logger.info(f'Parallel execution: {"enabled" if enabled else "disabled (sequential)"}')

    def _register_analysis_methods(self):
        return {
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
    
    def set_dump_folder(self, dump_folder: str):
        if not os.path.exists(dump_folder) or not os.path.isdir(dump_folder):
            self.logger.error(f'Dump folder does not exist or is not a directory: {dump_folder}')
            sys.exit(1)
        
        if 'analysis' not in self.config:
            self.config['analysis'] = {}

        self.config['analysis']['dump_folder'] = dump_folder
        self.logger.info(f'Dump folder set to: {dump_folder}')
        return True
 
    def set_output_folder(self, output_folder: str = None) -> str:
        output_folder = self.yaml_config.analysis_output_path
        self.logger.info(f'Using YamlConfig analysis output path: {output_folder}')

        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        if 'analysis' not in self.config:
            self.config['analysis'] = {}
            
        self.config['analysis']['output_folder'] = output_folder
        self.logger.info(f'Analysis output folder set to: {output_folder}')
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
        
        # Determine which analyses to run
        analyses_to_run = self._get_analyses_to_run(analysis_type)
        if not analyses_to_run:
            os.chdir(original_dir)
            return False

        success = True
        if self.parallel_execution and len(analyses_to_run) > 1:
            success = self.run_analysis_parallel(analyses_to_run, timestep)
        else:
            success = self._run_analyses_sequential(analyses_to_run, timestep)

        os.chdir(original_dir)
        elapsed_time = time.time() - start_time
        self.logger.info(f'Analysis completed in {elapsed_time:.2f} seconds')
        self.logger.info(f'Analysis results saved to {output_folder}')
        
        return success

    def _run_analyses_sequential(self, analyses_to_run, timestep):
        success = True
        for name, function in analyses_to_run:
            try:
                self.logger.info(f'Starting "{name}" analysis...')
                result = function(timestep)
                if result:
                    self.logger.info(f'Completed "{name}" analysis')
                else:
                    self.logger.error(f'"{name}" analysis reported failure')
                    success = False
                if self.memory_efficient:
                    self.parser.clear_data_cache()
                    gc.collect()
            except Exception as e:
                self.logger.error(f'Error in "{name}" analysis: {e}')
                success = False
        return success

    def run_analysis_parallel(self, analyses_to_run, timestep):
        success = True
        max_workers = min(len(analyses_to_run), self.max_workers)
        self.logger.info(f'Running {len(analyses_to_run)} analyses in parallel with {max_workers} workers')
        if self.memory_efficient and len(analyses_to_run) > max_workers:
            batch_size = max(1, max_workers // 2)
            self.logger.info(f'Memory efficient mode: processing in batches of {batch_size}')
            for i in range(0, len(analyses_to_run), batch_size):
                batch = analyses_to_run[i:i + batch_size]
                batch_success = self._execute_parallel_batch(batch, timestep, max_workers)
                success = success and batch_success
                self.parser.clear_data_cache()
                gc.collect()
        else:
            success = self._execute_parallel_batch(analyses_to_run, timestep, max_workers)
        return success

    def _execute_parallel_batch(self, analyses_batch, timestep, max_workers):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(function, timestep): name
                for name, function in analyses_batch
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.logger.info(f'Completed "{name}" analysis')
                    else:
                        self.logger.error(f'"{name}" analysis reported failure')
                        success = False    
                except Exception as e:
                    self.logger.error(f'Error in "{name}" analysis: {e}')
                    success = False
        return success
    
    
    def _get_analyses_to_run(self, analysis_type):
        if not analysis_type:
            return list(self.analysis_registry.items())
        
        if analysis_type in self.analysis_registry:
            return [(analysis_type, self.analysis_registry[analysis_type])]

        self.logger.error(f'Unknown analysis type: {analysis_type}')
        self.logger.info(f'Available types: {", ".join(self.analysis_registry.keys())}')
        return []
    
    def _execute_analyses(self, analyses, timestep):
        success = True
        max_workers = min(len(analyses), os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(function, timestep): name
                for name, function in analyses
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.logger.info(f'Completed "{name}" analysis')
                    else:
                        self.logger.error(f'"{name}" analysis reported failure')
                        success = False 
                except Exception as e:
                    self.logger.error(f'Error in "{name}" analysis: {e}')
                    success = False
        return success
                    
    def run_cna_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing CNA analysis')
        
        visualizer = visualizers.CommonNeighborAnalysisVisualizer(self.parser)

        self.logger.info('Generating CNA distribution plot')
        visualizer.plot_structure_distribution(timestep)

        self.logger.info('Generating CNA evolution plot')
        visualizer.plot_structure_evolution()

        self.logger.info('Generating CNA spatial distribution heatmap')
        visualizer.plot_structure_heatmap(timestep)

        self.logger.info('Generating CNA structure comparison')
        visualizer.plot_structure_comparison(0, timestep)

        return True

    def run_coordination_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing Coordination analysis')
        
        visualizer = visualizers.CoordinationVisualizer(self.parser)

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
    
    def run_debris_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing Debris analysis')
        
        visualizer = visualizers.DebrisVisualizer(self.parser)

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

    def run_hotspot_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing Hotspot analysis')

        visualizer = visualizers.HotspotVisualizer(self.parser)

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

    def run_vonmises_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing von Mises analysis')
        
        visualizer = visualizers.VonmisesVisualizer(self.parser)

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

    def run_centro_symmetric_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing Centro-Symmetric analysis')
        
        visualizer = visualizers.CentroSymmetricVisualizer(self.parser)

        self.logger.info('Generating Centro-Symmetric parameter distribution')
        visualizer.plot_centro_symmetric_distribution(timestep)
        visualizer.plot_centro_symmetric_distribution(timestep, log_scale=True)

        timesteps = self.parser.get_timesteps()
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

    def run_velocity_squared_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing Velocity Squared analysis')
        
        visualizer = visualizers.VelocitySquaredVisualizer(self.parser)

        self.logger.info('Generating temperature distribution plot')
        visualizer.plot_temperature_distribution(timestep)
        
        timesteps = self.parser.get_timesteps()
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

    def run_energy_analysis(self, timestep: int = -1) -> bool:
        self.logger.info('Initializing Energy analysis')
            
        visualizer = visualizers.EnergyVisualizer(self.parser)
        
        self.logger.info('Generating energy distribution plots')
        visualizer.plot_energy_distribution(timestep, energy_type='kinetic')
        visualizer.plot_energy_distribution(timestep, energy_type='potential')
        visualizer.plot_energy_distribution(timestep, energy_type='total')
        
        timesteps = self.parser.get_timesteps()
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