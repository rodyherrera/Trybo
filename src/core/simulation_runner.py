from typing import Optional, Tuple, Dict, Union, List

import os
import sys
import subprocess
import datetime
import re
import logging

class SimulationRunner:
    def __init__(
        self,
        simulation_file: str,
        lammps_executable = './lammps/build/lmp',
        build_script: str = 'toolchain/build_lammps_gpu.sh',
        log_level: int = logging.INFO
    ):
        self.logger = logging.getLogger('SimulationRunner')
        self._setup_logging(log_level)
        
        # Store configuration
        self.simulation_file = simulation_file
        self.lammps_executable = lammps_executable
        self.build_script = build_script
        
        # Print header
        current_date = datetime.datetime.now().strftime('%B %d %Y')
        self.logger.info(f'Trybo ({current_date} - Development - 1.0)')
        self.logger.info('Your suite of atomic-level analyses.')
    
    def _setup_logging(self, log_level: int):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
    
    def check_lammps_executable(self) -> bool:
        if os.path.isfile(self.lammps_executable):
            self.logger.info(f'LAMMPS executable found at {self.lammps_executable}')
            return True
        
        self.logger.warning(f'LAMMPS executable not found at {self.lammps_executable}')
        self.logger.info('Building LAMMPS automatically...')
        
        try:
            build_result = subprocess.run(['bash', self.build_script], check=False)
            
            if build_result.returncode == 0 and os.path.isfile(self.lammps_executable):
                self.logger.info('LAMMPS built successfully. Continuing...')
                return True
            self.logger.error('LAMMPS build failed. Please check the build output.')
            return False
                
        except Exception as e:
            self.logger.error(f'Error running build script: {str(e)}')
            return False
    
    def run_simulation(self) -> bool:
        self.logger.info('Starting simulation...')

        cmd = [
            'mpirun',
            '-np', '1',
            self.lammps_executable,
            '-sf', 'gpu',
            '-pk', 'gpu', '1', 'neigh', 'yes', 'split', '1.0', 'gpuID', '0',
            '-in', self.simulation_file
        ]
        
        self.logger.debug(f'Running command: {" ".join(cmd)}')
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            self.logger.info('Simulation completed successfully!')
            return True
        return False

    def check_simulation_file(self) -> bool:
        if os.path.isfile(self.simulation_file):
            self.logger.info(f'Simulation file found at {self.simulation_file}')
            return True
        self.logger.error(f'Error: Input file not found at {self.simulation_file}')
        return False

    def execute(self) -> bool:
        # Check LAMMPS executable and simulation file
        if not self.check_lammps_executable() or not self.check_simulation_file():
            return False
        
        return self.run_simulation()