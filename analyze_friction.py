import os
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
from ovito.io import import_file
from ovito.modifiers import CommonNeighborAnalysisModifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    '''Configuration parameters for friction analysis.'''
    logfile: str = 'log.lammps'
    time_step: float = 0.0005
    output_file: str = 'friction_coefficient.png'
    dpi: int = 300
    figure_size: Tuple[int, int] = (12, 7)
    show_plot: bool = False
    moving_avg_window: Optional[int] = None
    cf_column_patterns: List[str] = None

    def __post_init__(self):
        if self.cf_column_patterns is None:
            self.cf_column_patterns = [
                'v_coefficient_friction', 
                'c_coefficient', 
                'friction'
            ]
class LammpsLogParser:
    '''Parser for LAMMPS log files to extract data sections.'''
    def __init__(self, logfile: Union[str, Path]):
        self.logfile = Path(logfile)    
        self._validate_file()

    def _validate_file(self) -> None:
        if not self.logfile.exists():
            raise FileNotFoundError(f'Log file not found: {self.logfile}')
        if not self.logfile.is_file():
            raise ValueError(f'Path is not a file: {self.logfile}')
        if not os.access(self.logfile, os.R_OK):
            raise PermissionError(f'No permission to read file: {self.logfile}')
        
    def extract_data_sections(self) -> List[Tuple[List[str], np.ndarray]]:
        logger.info(f'Processing log file: {self.logfile}')
        step_pattern = re.compile(r'Step\s+')
        data_sections = []
        headers = None
        current_section = []
        try:
            with open(self.logfile, 'r') as file:
                in_data_section = False
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    # Detect table headers
                    if step_pattern.match(line):
                        headers = line.split()
                        in_data_section = True
                        current_section = []
                        logger.debug(f'Found data section at line {line_num} with headers: {headers}')
                        continue
                    # Process data section
                    if in_data_section:
                        if line and all(self._is_numeric_value(value) for value in line.split()):
                            current_section.append(line.split())
                        else:
                            if current_section:
                                data_array = np.array(current_section, dtype=float)
                                data_sections.append((headers, data_array))
                                logger.debug(f'Completed data section with {len(current_section)} rows')
                            in_data_section = False
                if in_data_section and current_section:
                    data_array = np.array(current_section, dtype=float)
                    data_sections.append((headers, data_array))
        except Exception as e:
            logger.error(f'Error parsing log file: {str(e)}')
            raise

        if not data_sections:
            logger.warning('No data sections found in the log file')

        return data_sections
    
    @staticmethod
    def _is_numeric_value(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False
        
class FrictionAnalyzer:
    '''Analyzes friction coefficient data from LAMMPS log files.'''
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.parser= LammpsLogParser(self.config.logfile)

    def analyze(self) -> Dict[str, Any]:
        logger.info('Starting friction coefficient analysis')
        # Extract data from log file
        data_sections = self.parser.extract_data_sections()
        if not data_sections:
            raise ValueError('No data sections found in log file')
        # Find the largest data section (probably the main simulation)
        headers, data = max(data_sections, key=lambda x: len(x[1]))
        df = pd.DataFrame(data, columns=headers)
        step_column = 'Step'
        if step_column not in headers:
            raise ValueError(f'Step column "{step_column}" not found in data')
        # Find friction coefficient column
        cf_column = self._find_cf_column(headers)
        if not cf_column:
            column_list = ', '.join(headers)
            raise ValueError(
                f'No friction coefficient column found.'
                f'Available columns: {column_list}'
            )
        logger.info(f'Using column "{cf_column}" for friction coefficient')
        df['Time'] = df[step_column].astype(float) * self.config.time_step
        # Apply moving average if configured
        if self.config.moving_avg_window:
            window = self.config.moving_avg_window
            cf_plot_column = f'{cf_column}_smoothed'
            df[cf_plot_column] = df[cf_column].rolling(window=window, center=True).mean()
            logger.info(f'Applied {window}-point moving average')
        else:
            cf_plot_column = cf_column

        fig, ax = self._create_plot(df, cf_plot_column)

        output_path = self.config.output_file
        fig.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f'Plot saved to {output_path}')

        if self.config.show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return {
            'data': df,
            'friction_column': cf_column,
            'mean_friction': df[cf_column].mean(),
            'std_friction': df[cf_column].std(),
            'time_range': (df['Time'].min(), df['Time'].max()),
            'output_path': output_path
        }
    
    def _find_cf_column(self, headers: List[str]) -> Optional[str]:
        for pattern in self.config.cf_column_patterns:
            matches = [header for header in headers if pattern in header.lower()]
            if matches:
                return matches[0]
        return None
    
    def _create_plot(self, df: pd.DataFrame, cf_column: str) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        # Plot friction coeffiecient
        ax.plot(df['Time'], df[cf_column], 'r--', linewidth=2, alpha=0.8)
        # Add mean value line
        mean_value = df[cf_column].mean()
        ax.axhline(y=mean_value, color='k', linestyle='--', alpha=0.7, label=f'Mean: {mean_value:.3f}')
        # Calculate standard deviation
        std_value = df[cf_column].std()
        
        ax.set_xlabel('Time (ps)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coefficient of Friction', fontsize=14, fontweight='bold')
        ax.set_title('Friction Coefficient Evolution', fontsize=16, fontweight='bold')
        
        stats_text = (
            f'Mean: {mean_value:.3f}\n'
            f'Std Dev: {std_value:.3f}\n'
            f'Range: [{df[cf_column].min():.3f}, {df[cf_column].max():.3f}]'
        )

        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round, pad=0.5', facecolor='white', alpha=0.7))

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)

        plt.tight_layout()

        return fig, ax
    
def analyze_structure(dump_file: str, output_prefix: str = 'structure_analysis'):
    try:
        logger.info(f'Importing dump file: {dump_file}')
        pipeline = import_file(dump_file)

        # Apply common neighbor analysis
        logger.info('Applying common neighbor analysis')
        pipeline.modifiers.append(CommonNeighborAnalysisModifier(
            mode=CommonNeighborAnalysisModifier.Mode.ConstructDislocations
        ))
        data = pipeline.compute()

        structure_types = data.particles.structure_types
        if structure_types is not None:
            type_mapping = {
                0: 'Other',
                1: 'FCC',
                2: 'HCP',
                3: 'BCC',
                4: 'ICO'
            }
            
            unique, counts = np.unique(structure_types, return_counts=True)
            total_atoms = len(structure_types)

            fig, ax = plt.subplots(figsize=(10, 6))

            labels = [type_mapping.get(type, f'Type {type}') for type in unique]
            percentages = counts / total_atoms * 100
            bars = ax.bar(labels, percentages, color=['gray', 'green', 'red', 'blue', 'purple'])

            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f'{percentage:.1f}%',
                    ha='center', 
                    va='bottom', 
                    fontsize=10
                )
                    
            ax.set_xlabel('Crystal Structure', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage of Atoms (%)', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Crystal Structures', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            output_file = f'{output_prefix}_crystal_structure.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f'Crystal structure analysis saved to {output_file}')
            return output_file
        else:
            logger.warning('No structure type data found in dump file')
            return None
    except Exception as e:
        logger.error(f'Error in structure analysis: {str(e)}')
        return None
        
def main():
    parser = argparse.ArgumentParser(description='LAMMPS Friction Coefficient Analyzer')
    parser.add_argument('--logfile', default='log.lammps', 
                        help='Path to LAMMPS log file')
    parser.add_argument('--time-step', type=float, default=0.0005,
                        help='Time step in ps')
    parser.add_argument('--output', default='friction_coefficient.png', 
                        help='Output image file path')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output image')
    parser.add_argument('--smooth', type=int, default=None,
                        help='Apply moving average with specified window size')
    parser.add_argument('--show', action='store_true',
                        help='Show plot in addition to saving')
    parser.add_argument('--dump-file', default=None,
                        help='Optional LAMMPS dump file for structure analysis')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = AnalysisConfig(
        logfile=args.logfile,
        time_step=args.time_step,
        output_file=args.output,
        dpi=args.dpi,
        show_plot=args.show,
        moving_avg_window=args.smooth
    )
    
    try:
        analyzer = FrictionAnalyzer(config)
        results = analyzer.analyze()
        print('\nFriction Analysis Results')
        print(f'Mean coefficient: {results['mean_friction']:.4f}')
        print(f'Standard deviation: {results['std_friction']:.4f}')
        print(f'Time range: [{results['time_range'][0]:.2f}, {results['time_range'][1]:.2f}] ps')
        print(f'Plot saved to: {results['output_path']}')
        if args.dump_file:
            output_file = analyze_structure(args.dump_file)
            if output_file:
                print(f'Structure analysis saved to: {output_file}')
        
    except Exception as e:
        logger.error(f'Analysis failed: {str(e)}')
        if args.debug:
            raise
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())