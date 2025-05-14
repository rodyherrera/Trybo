from typing import Dict, List
import os
import subprocess
import datetime
import re
import logging
import time

class SimulationRunner:
    def __init__(
        self,
        simulation_file: str,
        lammps_executable = './lammps/build/lmp',
        log_level = logging.INFO,
        optimize_input = True
    ):
        '''
        Initialize the SimulationRunner with the specified parameters.
        
        Args:
            simulation_file: Path to the LAMMPS input file
            lammps_executable: Path to the LAMMPS executable
            log_level: Logging level for output verbosity
            profile: Whether to enable performance profiling
            optimize_input: Whether to automatically optimize the input file
        '''
        self.logger = logging.getLogger('SimulationRunner')
        self._setup_logging(log_level)
        
        self.simulation_file = simulation_file
        self.lammps_executable = lammps_executable
        self.optimize_input = optimize_input
        
        self.start_time = 0
        self.end_time = 0
        
        current_date = datetime.datetime.now().strftime('%B %d %Y')
        self.logger.info(f'Trybo ({current_date} - Development - 1.0)')
        self.logger.info('Your suite of atomic-level analyses.')
        
    def _setup_logging(self, log_level):
        '''
        Configure the logging system with appropriate format and level.
        
        Args:
            log_level: The logging level to use
        '''
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
    
    def check_lammps(self) -> bool:
        '''
        Verify that the LAMMPS executable exists and is accessible.
        
        Returns:
            bool: True if the executable was found, False otherwise
        '''
        if os.path.isfile(self.lammps_executable):
            self.logger.info(f'LAMMPS executable found at {self.lammps_executable}')
            return True
            
        self.logger.error(f'LAMMPS executable not found at {self.lammps_executable}')
        return False
    
    def check_input_file(self) -> bool:
        '''
        Verify that the input file exists and is accessible.
        
        Returns:
            bool: True if the input file was found, False otherwise
        '''
        if os.path.isfile(self.simulation_file):
            self.logger.info(f'Simulation file found at {self.simulation_file}')
            return True
            
        self.logger.error(f'Input file not found at {self.simulation_file}')
        return False
    
    def detect_gpu(self) -> Dict:
        '''
        Get detailed information about available GPUs.
        
        Uses nvidia-smi to retrieve GPU details including name, memory,
        and compute capability. Falls back to default values if detection fails.
        
        Returns:
            Dict: Dictionary containing GPU information
        '''
        gpu_info = {
            'name': 'Unknown NVIDIA GPU',
            'memory': 8000,
            'is_rtx': False,
            'compute_capability': 0.0,
            'cores': 0
        }
        
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,compute_capability', '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=3
        )
        
        if result.returncode == 0 and result.stdout:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 1:
                gpu_info['name'] = parts[0].strip()
                
            if len(parts) >= 2:
                memory_match = re.search(r'(\d+)', parts[1])
                if memory_match:
                    gpu_info['memory'] = int(memory_match.group(1))
            
            if len(parts) >= 3:
                cc_match = re.search(r'(\d+\.\d+)', parts[2])
                if cc_match:
                    gpu_info['compute_capability'] = float(cc_match.group(1))
            
            if 'RTX' in gpu_info['name']:
                gpu_info['is_rtx'] = True
                
            if 'RTX 3090' in gpu_info['name']:
                gpu_info['cores'] = 10496
            elif 'RTX 3080' in gpu_info['name']:
                gpu_info['cores'] = 8704
            elif 'RTX 2080' in gpu_info['name']:
                gpu_info['cores'] = 2944
            elif 'GTX 1080' in gpu_info['name']:
                gpu_info['cores'] = 2560
                
        self.logger.info(f'GPU detected: {gpu_info["name"]} with {gpu_info["memory"]}MB memory')
        if gpu_info['compute_capability'] > 0:
            self.logger.info(f'Compute capability: {gpu_info["compute_capability"]}')
        
        return gpu_info
    
    def _calculate_optimal_parameters(self, gpu_info: Dict) -> Dict:
        '''
        Calculate optimal parameters based on GPU capabilities.
        
        Determines the best settings for GPU performance based on the
        detected hardware and the size of the simulation.
        
        Args:
            gpu_info: Dictionary containing GPU information
            
        Returns:
            Dict: Dictionary with optimized parameters
        '''
        params = {
            'tpa': 256,
            'binsize': 8.0,
            'split': 1.0,
            'mpi_processes': 1
        }
        
        cores = os.cpu_count() or 4
        
        if gpu_info['is_rtx'] and gpu_info['memory'] >= 16000:
            params['tpa'] = 512
        elif gpu_info['memory'] < 6000 or 'GTX' in gpu_info['name']:
            params['tpa'] = 128
        
        if gpu_info['memory'] >= 16000:
            params['binsize'] = 12.0
        elif gpu_info['memory'] < 8000:
            params['binsize'] = 6.0
        
        if cores >= 16:
            params['mpi_processes'] = 4
        elif cores >= 8:
            params['mpi_processes'] = 2
        
        analysis = self.analyze_input_file()
        if analysis['size'] == 'large':
            params['split'] = 0.8
        elif analysis['size'] == 'small':
            params['split'] = 1.0
        
        self.logger.info(f'Optimal parameters calculated:')
        self.logger.info(f'  - TPA: {params["tpa"]}')
        self.logger.info(f'  - Binsize: {params["binsize"]}')
        self.logger.info(f'  - Split: {params["split"]}')
        self.logger.info(f'  - MPI processes: {params["mpi_processes"]}')
        
        return params
    
    def analyze_input_file(self) -> Dict:
        '''
        Analyze input file to determine optimization opportunities.
        
        Parses the LAMMPS input file to extract key settings such as timestep,
        pair style, neighbor settings, run length, and system size estimation.
        
        Returns:
            Dict: Analysis results containing identified settings
        '''
        analysis = {
            'size': 'medium',
            'timestep': 0.001,
            'pair_style': 'unknown',
            'neighbor_settings': {},
            'run_steps': 0
        }
        
        try:
            with open(self.simulation_file, 'r') as f:
                content = f.read()
            
            timestep_match = re.search(r'timestep\s+(\S+)', content)
            if timestep_match:
                analysis['timestep'] = float(timestep_match.group(1))
            
            pair_style_match = re.search(r'pair_style\s+(\S+)', content)
            if pair_style_match:
                analysis['pair_style'] = pair_style_match.group(1)
            
            neigh_skin_match = re.search(r'neighbor\s+(\S+)\s+(\S+)', content)
            if neigh_skin_match:
                analysis['neighbor_settings']['skin'] = float(neigh_skin_match.group(1))
                analysis['neighbor_settings']['style'] = neigh_skin_match.group(2)
            
            run_match = re.search(r'run\s+(\d+)', content)
            if run_match:
                analysis['run_steps'] = int(run_match.group(1))
            
            atoms_match = re.search(r'create_atoms\s+\d+\s+\w+\s+(\d+)', content)
            if atoms_match:
                atoms = int(atoms_match.group(1))
                if atoms < 10000:
                    analysis['size'] = 'small'
                elif atoms < 100000:
                    analysis['size'] = 'medium'
                else:
                    analysis['size'] = 'large'
            
            region_match = re.search(r'region\s+\w+\s+\w+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', content)
            if region_match and not atoms_match:
                try:
                    x1, x2 = float(region_match.group(1)), float(region_match.group(2))
                    y1, y2 = float(region_match.group(3)), float(region_match.group(4))
                    z1, z2 = float(region_match.group(5)), float(region_match.group(6))
                    
                    vol = abs((x2-x1) * (y2-y1) * (z2-z1))
                    if vol < 1000:
                        analysis['size'] = 'small'
                    elif vol < 100000:
                        analysis['size'] = 'medium'
                    else:
                        analysis['size'] = 'large'
                except:
                    pass
            
            return analysis
        except Exception as e:
            self.logger.warning(f'Could not analyze input file: {str(e)}')
            return analysis
    
    def optimize_input_file(self, analysis: Dict) -> str:
        '''
        Create an optimized version of the input file based on analysis.
        
        Modifies the original input file to add performance optimizations
        based on analysis results and detected hardware capabilities.
        
        Args:
            analysis: Dictionary containing input file analysis results
            
        Returns:
            str: Path to the optimized input file, or the original if optimization fails
        '''
        if not os.path.exists(self.simulation_file) or not self.optimize_input:
            return self.simulation_file
        
        try:
            with open(self.simulation_file, 'r') as f:
                content = f.readlines()
            
            optimized_file = f"{os.path.splitext(self.simulation_file)[0]}_optimized.in"
            with open(optimized_file, 'w') as f:
                f.write('# Optimized input file generated by SimulationRunner\n')
                
                neighbor_added = False
                thermo_added = False
                balance_added = False
                
                for line in content:
                    line_stripped = line.strip()
                    if line_stripped.startswith('neighbor '):
                        neighbor_added = True
                        gpu_info = self.detect_gpu()
                        if gpu_info['memory'] >= 8000:
                            f.write('\nneighbor 2.0 bin\n')
                        else:
                            f.write(line)
                    elif line_stripped.startswith('thermo '):
                        thermo_added = True
                        if analysis['run_steps'] > 10000:
                            freq = max(100, analysis['run_steps'] // 100)
                            f.write(f'thermo {freq}\n')
                        else:
                            f.write(line)
                    elif analysis['size'] == 'large' and line_stripped.startswith('run ') and not balance_added:
                        balance_added = True
                        f.write('balance 1.0 rcb\n')
                        f.write(line)
                    else:
                        f.write(line)
                
                if not neighbor_added:
                    f.write('\nneighbor 2.0 bin\n')
                
                if not thermo_added:
                    freq = max(100, analysis['run_steps'] // 100) if analysis['run_steps'] > 0 else 100
                    f.write(f'thermo {freq}\n')
            
            self.logger.info(f'Created optimized input file: {optimized_file}')
            return optimized_file
        except Exception as e:
            self.logger.warning(f'Could not optimize input file: {str(e)}')
            return self.simulation_file
    
    def _check_gpu_compatibility(self) -> bool:
        '''
        Check if the input file contains pair styles compatible with GPU acceleration.
        
        Analyzes the LAMMPS input file to determine if the pair style used
        is compatible with the GPU package for acceleration.
        
        Returns:
            bool: True if compatible pair style is found, False otherwise
        '''
        try:
            with open(self.simulation_file, 'r') as f:
                content = f.read()
                
            gpu_styles = ['lj/cut', 'gayberne', 'lj/charmm', 'eam', 'morse', 'buck', 'table']
            
            pair_style_match = re.search(r'pair_style\s+(\S+)', content)
            if pair_style_match:
                style = pair_style_match.group(1)
                for gpu_style in gpu_styles:
                    if gpu_style in style:
                        self.logger.info(f'Detected GPU-compatible pair style: {style}')
                        return True
            
            self.logger.warning('No GPU-compatible pair style detected, some optimizations may not be effective')
            return False
        except Exception as e:
            self.logger.warning(f'Could not analyze input file: {str(e)}')
            return True
    
    def _check_command_exists(self, command):
        '''
        Check if a command exists on the system.
        
        Args:
            command: Name of the command to check
            
        Returns:
            bool: True if the command exists, False otherwise
        '''
        try:
            proc = subprocess.run(['which', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return proc.returncode == 0
        except:
            return False
    
    def run_simulation(self) -> bool:
        '''
        Run the simulation with optimized parameters.
        
        Builds a command with optimized parameters based on detected hardware
        and input file analysis, then executes the LAMMPS simulation.
        
        Returns:
            bool: True if simulation completed successfully, False otherwise
        '''
        self.logger.info('Starting simulation...')
        
        gpu_info = self.detect_gpu()
        params = self._calculate_optimal_parameters(gpu_info)
        
        gpu_compatible = self._check_gpu_compatibility()
        
        cmd = []
        
        if params['mpi_processes'] > 1 and self._check_command_exists('mpirun'):
            cmd = ['mpirun', '-np', str(params['mpi_processes'])]
            
            if self._check_command_exists('numactl'):
                cmd = ['numactl', '--localalloc'] + cmd
        
        cmd.append(self.lammps_executable)
        
        if gpu_compatible:
            cmd.extend([
                '-sf', 'gpu',
                '-pk', 'gpu', '1', 'split', str(params['split']), 'binsize', str(params['binsize']), 'tpa', str(params['tpa']), 'gpuID', '0'
            ])

        
        cmd.extend(['-in', self.simulation_file])
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        if params['mpi_processes'] > 1:
            env['CUDA_LAUNCH_BLOCKING'] = '0'
        
        cores_per_mpi = max(1, (os.cpu_count() or 4) // params['mpi_processes'])
        env['OMP_NUM_THREADS'] = str(cores_per_mpi)
        
        if self._check_command_exists('taskset'):
            cmd = ['taskset', '-c', '0-' + str(os.cpu_count() or 4 - 1)] + cmd
        
        self.logger.info(f'Running with optimized parameters')
        
        try:
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            
            self.start_time = time.time()
            
            result = subprocess.run(cmd, env=env, check=False)
            
            self.end_time = time.time()
            
            elapsed = self.end_time - self.start_time
            self.logger.info(f'Simulation runtime: {elapsed:.2f} seconds')
            
            if result.returncode == 0:
                self.logger.info('Simulation completed successfully!')
                return True
            else:
                self.logger.error(f'Simulation failed with exit code {result.returncode}')
                return self._run_fallback()
        except Exception as e:
            self.logger.error(f'Error running simulation: {str(e)}')
            return False
    
    def _run_fallback(self) -> bool:
        '''
        Try alternative simulation methods if the optimized approach fails.
        
        Attempts to run the simulation with progressively simpler parameters
        if the initial optimized run fails.
        
        Returns:
            bool: True if any fallback method succeeds, False otherwise
        '''
        self.logger.info('Trying fallback with simpler parameters...')
        
        cmd = [
            self.lammps_executable,
            '-sf', 'gpu',
            '-pk', 'gpu', '1',
            '-in', self.simulation_file
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        try:
            result = subprocess.run(cmd, env=env, check=False)
            
            if result.returncode == 0:
                self.logger.info('Fallback simulation completed successfully!')
                return True
                
            self.logger.info('Trying second fallback without GPU acceleration...')
            cmd = [self.lammps_executable, '-in', self.simulation_file]
            result = subprocess.run(cmd, check=False)
            
            return result.returncode == 0
        except:
            return False
    
    def _analyze_profiling_results(self):
        '''
        Analyze log file for performance bottlenecks.
        
        Examines the LAMMPS log file to extract performance metrics
        and identify potential bottlenecks in the simulation.
        '''
        log_file = 'log.lammps'
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                timing_section = re.search(r'Performance: (.*?)Loop time of([^#]*)', log_content, re.DOTALL)
                if timing_section:
                    self.logger.info('Performance Analysis:')
                    self.logger.info(timing_section.group(0))
                    
                    mflops_match = re.search(r'(\d+\.\d+) MFlops', log_content)
                    if mflops_match:
                        mflops = float(mflops_match.group(1))
                        if mflops < 100:
                            self.logger.warning('Low MFlops detected. Consider increasing TPA or using a more powerful GPU')
                    
                    pair_match = re.search(r'Pair\s+:\s+(\d+\.\d+)', log_content)
                    neigh_match = re.search(r'Neigh\s+:\s+(\d+\.\d+)', log_content)
                    comm_match = re.search(r'Comm\s+:\s+(\d+\.\d+)', log_content)
                    
                    if pair_match and neigh_match and comm_match:
                        pair_time = float(pair_match.group(1))
                        neigh_time = float(neigh_match.group(1))
                        comm_time = float(comm_match.group(1))
                        
                        if neigh_time / pair_time > 0.5:
                            self.logger.warning('Neighbor list calculation taking significant time. Consider adjusting neighbor skin distance.')
                        
                        if comm_time / pair_time > 0.3:
                            self.logger.warning('Communication overhead is high. Consider reducing MPI processes or optimizing domain decomposition.')
        except Exception as e:
            self.logger.warning(f'Could not analyze performance results: {str(e)}')
    
    def execute(self) -> bool:
        '''
        Execute the complete simulation workflow.
        
        This is the main entry point that coordinates the entire simulation
        process, including input file checking, optimization, execution,
        and performance analysis.
        
        Returns:
            bool: True if simulation completed successfully, False otherwise
        '''
        try:
            if not self.check_lammps() or not self.check_input_file():
                return False
            
            analysis = self.analyze_input_file()
            optimized_input = self.optimize_input_file(analysis)
            
            original_input = self.simulation_file
            if optimized_input != original_input:
                self.simulation_file = optimized_input
                
            success = self.run_simulation()
            
            self._analyze_profiling_results()
            
            self.simulation_file = original_input
            
            return success
        except KeyboardInterrupt:
            self.logger.warning('Simulation interrupted by user')
            return False
        except Exception as e:
            self.logger.error(f'Unexpected error: {str(e)}')
            return False
    
        except Exception as e:
            self.logger.error(f'Error in profiling: {str(e)}')
            return self.run_simulation()