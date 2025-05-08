from core.yaml_config import YamlConfig, BUILDS_DIR
from core.analyzer import Analyzer
import sys

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 main.py <yaml_config_file> [output_directory]')
        sys.exit(1)
    yaml_file = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else BUILDS_DIR
    yaml_config = YamlConfig(yaml_file, output_directory)
    if yaml_config.load_config() and yaml_config.render_template():
        print('Generation completed successfully.')
        sys.exit(0)
    print('Generation failed.')
    sys.exit(1)

if __name__ == '__main__':
    main()