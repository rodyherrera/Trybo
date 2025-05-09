from core.yaml_config import YamlConfig, BUILDS_DIR
from core.analyzer import Analyzer
from core.simulation_runner import SimulationRunner
import sys

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 main.py <yaml_config_file> [output_directory]')
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else BUILDS_DIR
    
    # Create the YAML config handler
    yaml_config = YamlConfig(yaml_file, output_directory)
    
    # Load the configuration first
    if not yaml_config.load_config():
        print('Failed to load YAML configuration.')
        sys.exit(1)
    
    # Render the template to generate the simulation file
    if not yaml_config.render_template():
        print('Failed to render simulation template.')
        sys.exit(1)
    
    # Now get the simulation file path (after render_template has set output_filename)
    simulation_file = yaml_config.get_simulation_file_path()
    
    # Create simulation runner and analyzer
    simulation_runner = SimulationRunner(simulation_file)
    analyzer = Analyzer(yaml_file, yaml_config.output_directory)
    
    # Execute simulation
    if simulation_runner.execute():
        print('Simulation completed successfully.')
        analyzer.run_analysis()
        sys.exit(0)
    else:
        print('Simulation failed.')
        sys.exit(1)

if __name__ == '__main__':
    main()