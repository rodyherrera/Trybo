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
            'cna': self.run_cna_analysis
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