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