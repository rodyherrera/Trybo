from jinja2 import Environment, FileSystemLoader
from typing import Optional, Union
import os
import yaml
import datetime
import requests
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, '..', 'templates')
TEMPLATE_FILE = os.path.join(TEMPLATES_DIR, 'base', 'in.main')
BUILDS_DIR = os.path.join(TEMPLATES_DIR, 'builds')
POTENTIALS_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'potentials')

LAMMPS_BASE_POTENTIALS_URL = 'https://raw.githubusercontent.com/lammps/lammps/refs/heads/stable/potentials'

class YamlConfig:
    '''
    YamlConfig handles loading a YAML configuration file, ensuring necessary
    directories exist, fetching LAMMPS potentials if needed, and rendering
    a Jinja2 template to generate a LAMMPS input script.
    '''

    def __init__(self, yaml_file, output_directory):
        '''
        Initialize YamlConfig with paths for YAML input and output directory.

        Args:
            yaml_file: Path to the YAML configuration file.
            output_directory: Directory where generated files will be written.
        '''
        self.yaml_file = yaml_file
        self.output_directory = output_directory
        self.config = None
        self.output_filename = None
        self.analysis_output_path = None
        self.ensure_output_directories()

    def ensure_output_directories(self):
        '''
        Create required directories for templates, builds, potentials, and output.
        '''
        os.makedirs(os.path.dirname(TEMPLATE_FILE), exist_ok=True)
        os.makedirs(BUILDS_DIR, exist_ok=True)
        os.makedirs(POTENTIALS_DIR, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)

    def get_output_filename(self):
        '''
        Determine the LAMMPS script filename based on the 'simulation.name' field.
        Falls back to a default name if not specified.

        Returns:
            The generated output filename ending in '.lammps'.
        '''
        if self.config and 'simulation' in self.config and 'name' in self.config['simulation']:
            name = self.config['simulation']['name'].lower().replace(' ', '_')
            sanitized = ''.join(c for c in name if c.isalnum() or c in ['_', '-'])
            filename = f'{sanitized}.lammps'
        else:
            filename = 'generated_simulation.lammps'
        self.output_filename = filename
        return filename
    
    def load_config(self):
        '''
        Load the YAML configuration into memory.

        Returns:
            True if loading succeeds, False otherwise.
        '''
        if not os.path.exists(self.yaml_file):
            print(f'YAML configuration file not found: {self.yaml_file}')
            return False
        
        if not os.path.exists(TEMPLATE_FILE):
            print(f'Jinja2 template file not found: {TEMPLATE_FILE}')
            print(f'Create template file at: {TEMPLATE_FILE}')
            return False

        try:
            with open(self.yaml_file, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f'YAML file loaded successfully: {self.yaml_file}')
            return True
        except Exception as e:
            print(f'Error loading YAML file {self.yaml_file}: {e}')
            return False

    def create_analysis_output_path(self):
        '''
        Create the analysis output path with the structure:
        /project_root/analysis_results/[project_name]/[timestamp]/
        Returns the created path.
        '''
        # Get project name from filename (without extension)
        if not self.output_filename:
            self.get_output_filename()
            
        project_name = os.path.splitext(self.output_filename)[0]
        
        # Generate timestamp in ISO format
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(SCRIPT_DIR))
        
        # Create full path structure
        analysis_path = os.path.join(
            project_root, 
            'analysis_results', 
            project_name,
            timestamp
        )
        
        # Create the directory structure
        os.makedirs(analysis_path, exist_ok=True)
        
        # Store for later use
        self.analysis_output_path = analysis_path
        
        return analysis_path

    def _extract_filename_from_potential_path(self):
        '''
        Extract the basename of the potential file defined in the config.
        '''
        potential = self.config['system']['material']['potential']
        filename = os.path.basename(potential)
        return filename
    
    def _fetch_potential_from_lammps_repo(self):
        '''
        Download a potential file from the official LAMMPS repository.

        Raises:
            RuntimeError: If the potential cannot be found remotely.
        '''
        potential_filename = self._extract_filename_from_potential_path()
        print(f'Searching for the potential "{potential_filename}" in the LAMMPS repository...')

        req = requests.get(f'{LAMMPS_BASE_POTENTIALS_URL}/{potential_filename}')
        if req.status_code != 200:
            raise RuntimeError(f"The potential was not found in the repository. There's nothing I can do for you. Download the potential you want to use and specify the absolute path to the file.")
        
        potential_file_path = f'{POTENTIALS_DIR}/{potential_filename}'
        self.config['system']['material']['potential'] = potential_file_path

        with open(potential_file_path, 'wb') as potential_file:
            potential_file.write(req.content)

        print(f'Potential downloaded and saved to {potential_file_path} successfully.')
        
    def check_material_potential(self):
        '''
        Ensure the material potential path is valid, otherwise fetch it.
        '''
        potential = self.config['system']['material']['potential']
        if os.path.isfile(potential):
            print(f'Using {potential} potential provided from configuration.')
            return
        
        self._fetch_potential_from_lammps_repo()

    def render_template(self):
        '''
        Render the Jinja2 template using loaded config and write output files.

        Returns:
            True on success, False on failure.
        '''
        if not self.config:
            print('No configuration loaded. Call load_config() first.')
            return False
        
        self.check_material_potential()
        
        # Get the output filename first
        self.get_output_filename()
        
        # Create analysis output path
        analysis_path = self.create_analysis_output_path()
        
        if 'simulation' not in self.config:
            self.config['simulation'] = {}
            
        self.config['simulation']['output_directory'] = analysis_path
        print(f'Set output directory to: {analysis_path}')
        
        self.config['analysis'] = self.config.get('analysis', {})
        self.config['analysis']['dump_folder'] = analysis_path
        self.config['analysis']['output_folder'] = analysis_path
        
        try:
            # Jinja2 Environment
            template_directory = os.path.dirname(TEMPLATE_FILE)
            template_name = os.path.basename(TEMPLATE_FILE)
            environ = Environment(
                loader=FileSystemLoader(template_directory),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
            template = environ.get_template(template_name)
            output = template.render(**self.config)
            output_path = os.path.join(self.output_directory, self.output_filename)
            with open(output_path, 'w') as file:
                file.write(output)
            print(f'LAMMPS file successfully generated: {output_path}')
            
            # Also save the configuration to the analysis directory for reference
            config_backup_path = os.path.join(analysis_path, 'simulation_config.yml')
            with open(config_backup_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            print(f'Configuration saved to: {config_backup_path}')
            
            return True
        except Exception as e:
            print(f'Error rendering template: {str(e)}')
            return False
    
    def get_simulation_file_path(self) -> str:
        '''
        Return the absolute path to the generated LAMMPS script, validating safety.

        Returns:
            Absolute file path of the simulation script.

        Raises:
            ValueError: If the path is outside the intended output directory.
        '''
        # Make sure the output_filename is set
        if not self.output_filename:
            self.get_output_filename()
            
        if not self.output_filename:
            raise ValueError('Failed to generate an output filename. Make sure to load the config first.')
        
        # Combine directory and filename to get absolute path
        simulation_file_path = os.path.abspath(os.path.join(self.output_directory, self.output_filename))
        
        # Verify the path is within the intended output directory (security check)
        if not simulation_file_path.startswith(os.path.abspath(self.output_directory)):
            raise ValueError(f'Output path {simulation_file_path} is outside the specified output directory')
        
        return simulation_file_path
    
    def get_analysis_output_path(self) -> str:
        '''
        Return the analysis output directory, creating it if necessary.
        '''
        if not self.analysis_output_path:
            self.create_analysis_output_path()
        return self.analysis_output_path