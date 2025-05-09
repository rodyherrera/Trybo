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
        
        # Initialize CPU information
        self.total_processors = 0
        self.physical_cores = 0
        self.use_processors = 0
        
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
            else:
                self.logger.error('LAMMPS build failed. Please check the build output.')
                return False
                
        except Exception as e:
            self.logger.error(f'Error running build script: {str(e)}')
            return False

    def detect_cpu_info(self):
        self.total_processors = os.cpu_count()
        if os.path.isfile('/proc/cpuinfo'):
            # Linux-specific approach
            try:
                result = subprocess.run(
                    "grep 'cpu cores' /proc/cpuinfo | head -1 | awk '{print $4}'", 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    text=True, 
                    check=True
                )
                physical_cores = result.stdout.strip()
                if physical_cores:
                    self.physical_cores = int(physical_cores)
                else:
                    self.physical_cores = self.total_processors // 2
            except:
                self.physical_cores = self.total_processors // 2
        else:
            # Estimation for other OS
            self.physical_cores = self.total_processors // 2
        
        # Single-Core CPUs
        if self.physical_cores < 1:
            self.physical_cores = 1
        
        # Calculate processors to use (n-1 to leave one for system)
        self.use_processors = max(1, self.physical_cores - 1)
        self.logger.info(f'System has {self.total_processors} logical processors ({self.physical_cores} physical cores)')
        self.logger.info(f'Using {self.use_processors} processors for simulation')
        