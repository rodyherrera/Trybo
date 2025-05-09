from jinja2 import Environment, FileSystemLoader
import os
import sys
import yaml
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, '..', 'templates')
TEMPLATE_FILE = os.path.join(TEMPLATES_DIR, 'wear_template.lammps')
BUILDS_DIR = os.path.join(TEMPLATES_DIR, 'builds')

class YamlConfig:
    def __init__(self, yaml_file, output_directory):
        self.yaml_file = yaml_file
        self.output_directory = output_directory
        self.config = None
        self.output_filename = None
        self.analysis_output_path = None
        self.ensure_output_directories()

    def ensure_output_directories(self):
        # Create template and builds directories if they don't exists
        os.makedirs(os.path.dirname(TEMPLATE_FILE), exist_ok=True)
        os.makedirs(BUILDS_DIR, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)

    def get_output_filename(self):
        # Determine output filename from YAML
        if self.config and 'simulation' in self.config and 'name' in self.config['simulation']:
            # Convert name to a valid filename by removing characters and replacecing spaces with underscores
            output_filename = self.config['simulation']['name'].lower().replace(' ', '_')
            # Remove non-alphanumeric characters except undescore and hyphens
            output_filename = ''.join(character for character in output_filename if character.isalnum() or character in ['_', '-'])
            output_filename += '.lammps'
        else:
            # Default name if not defined in YAML
            output_filename = 'generated_simulation.lammps'
        self.output_filename = output_filename
        return self.output_filename
    
    def load_config(self):
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
        """
        Create the analysis output path with the structure:
        /project_root/analysis_results/[project_name]/[timestamp]/
        Returns the created path.
        """
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

    def render_template(self):
        if not self.config:
            print('No configuration loaded. Call load_config() first.')
            return False
        
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
    
    def get_analysis_output_path(self):
        if not self.analysis_output_path:
            self.create_analysis_output_path()
        return self.analysis_output_path