from abc import ABC
from typing import List, Dict, Any, Union, Optional
import numpy as np
import gc

class BaseParser(ABC):
    def __init__(self, filename: str):
        self.filename = filename

        self._timesteps = []
        self._data = []
        self._headers = []
        self._is_parsed = False
        # Para parseo selectivo
        self._timestep_positions = {}
        self._file_positions = []
        # Simulation box dimensions
        self._box_bounds = None
        # Metadata from the dump file
        self._metadata = {}

        # The x, y, and z values ​​in the LAMMPS dump files.
        # ITEM: ATOMS id type x y z [...]
        self._atoms_spatial_coordinates_indices = []

        # Mapping between analysis types and column indices
        self._analysis_column_map = {}
        
        # Indexing the file at initialization for better performance
        self._index_file()

    def _index_file(self):
        with open(self.filename, 'r') as file:
            position = file.tell()
            line = file.readline()
            
            while line:
                if 'ITEM: TIMESTEP' in line:
                    # Read the timestep number
                    timestep = int(file.readline().strip())
                    # Save the position where this timestep's data begins
                    self._timestep_positions[timestep] = position
                    self._timesteps.append(timestep)
                    
                    # Skip ahead to find atoms data
                    while line and 'ITEM: ATOMS' not in line:
                        line = file.readline()
                        if not line:
                            break
                    
                    if line and 'ITEM: ATOMS' in line:
                        # Save the headers once
                        if not self._headers:
                            self._headers = line.split()[2:]
                            self._parse_atoms_spatial_coordinates_indices()
                            self._create_analysis_column_map()
                            self._is_parsed = True
                
                position = file.tell()
                line = file.readline()
        
        if not self._headers:
            raise ValueError('No headers were found in the file. Check the format.')

    def _parse_atoms_spatial_coordinates_indices(self):
        try:
            x_idx = self._headers.index('x')
            y_idx = self._headers.index('y')
            z_idx = self._headers.index('z')
        except ValueError:
            print('WARNING: The headers do not contain "x", "y", or "z". This is a critical error, and nothing may work as expected. Set the expected positions (2, 3, and 4, respectively).')
            x_idx = 2
            y_idx = 3
            z_idx = 4
        self._atoms_spatial_coordinates_indices = [x_idx, y_idx, z_idx]

    def _load_all_timesteps(self):
        self._data = []
        for timestep in self._timesteps:
            self._data.append(self._load_timestep_data(timestep))
        return self._timesteps, self._data, self._headers

    def _load_timestep_data(self, timestep):
        if timestep not in self._timestep_positions:
            if isinstance(timestep, int) and timestep < 0:
                try:
                    actual_timestep = self._timesteps[timestep]
                    return self._load_timestep_data(actual_timestep)
                except IndexError:
                    raise ValueError(f'Invalid timestep index: {timestep}')
            else:
                raise ValueError(f'Timestep {timestep} not found in file')
        with open(self.filename, 'r') as file:
            file.seek(self._timestep_positions[timestep])
            # Skip the TIMESTEP line and timestep value
            file.readline()
            file.readline()
            # Skip to the ATOMS section
            line = file.readline()
            while line and 'ITEM: ATOMS' not in line:
                line = file.readline()
            if not line or 'ITEM: ATOMS' not in line:
                raise ValueError(f'Could not find ATOMS section for timestep {timestep}')
            atoms_data = []
            line = file.readline()
            while line and 'ITEM:' not in line:
                values = [float(value) for value in line.split()]
                atoms_data.append(values)
                line = file.readline()
            return np.array(atoms_data)

    def _create_analysis_column_map(self):
        analysis_headers = {
            'centro_symmetric': 'c_center_symmetric',
            'cna': 'c_cna',
            'vonmises': 'v_atoms_stress',
            'velocity_squared': 'v_velocity_squared_atom',
            'cluster': 'c_cluster',
            'ke_hotspots': 'c_ke_hotspots',
            'is_hotspot': 'v_is_hotspot',
            'coord': 'c_coord',
            'ptm': ['c_ptm[1]', 'c_ptm[2]', 'c_ptm[3]', 'c_ptm[4]', 'c_ptm[5]', 'c_ptm[6]']
        }

        for analysis_type, header in analysis_headers.items():
            if isinstance(header, list):
                indices = []
                for h in header:
                    if h in self._headers:
                        indices.append(self._headers.index(h))
                if indices:
                    self._analysis_column_map[analysis_type] = indices
            elif header in self._headers:
                self._analysis_column_map[analysis_type] = self._headers.index(header)
        print(f'Analysis column map created: {self._analysis_column_map}')
    
    def get_analysis_data(self, analysis_type: str, timestep_idx: int = -1) -> Union[np.ndarray, List[np.ndarray]]:
        if analysis_type not in self._analysis_column_map:
            raise ValueError(f'Analysis type "{analysis_type}" not found in headers: {self._headers}')
        
        column_idx = self._analysis_column_map[analysis_type]

        if timestep_idx < 0:
            timestep_idx = len(self._timesteps) + timestep_idx
        
        timestep_data = self._get_timestep_data(timestep_idx)

        if isinstance(column_idx, list):
            # Return multiple columns for array data
            return [timestep_data[:, i] for i in column_idx]
        
        return timestep_data[:, column_idx]
    
    def _get_timestep_data(self, timestep_idx):
        if timestep_idx < len(self._data):
            timestep_data = self._data[timestep_idx]
            if timestep_data is not None and len(timestep_data) > 0:
                return timestep_data
        
        timestep = self._timesteps[timestep_idx]
        timestep_data = self._load_timestep_data(timestep)

        while len(self._data) <= timestep_idx:
            self._data.append(None)
        
        self._data[timestep_idx] = timestep_data
        return timestep_data

    def clear_data_cache(self):
        self._data = []
        gc.collect()

    def get_atom_group_indices(self, data):
        '''
        Identifies atom indices for different groups based on their position along the Z axis
        
        Args:
            data: Data array to use instead of the parser's data

        Returns:
            dict: Dictionary with indices for each group (lower_plane, upper_plane, nanoparticle, all)
        '''
        # Get Z coordinates
        x, y, z = self.get_atoms_spatial_coordinates(data)
        # Calculate Z thresholds
        z_min = np.min(z)
        z_max = np.max(z)
        # Lower plane up to 2.5 Å
        z_threshold_lower = z_min + 2.5
        # Upper plane from z_max - 2.5 Å
        z_threshold_upper = z_max - 2.5
        # Identify groups based on Z position
        lower_plane_mask = z <= z_threshold_lower
        upper_plane_mask = z >= z_threshold_upper
        nanoparticle_mask = ~(lower_plane_mask | upper_plane_mask)
        # Return dictionary with indices for each group
        return {
            'lower_plane': np.where(lower_plane_mask)[0],
            'upper_plane': np.where(upper_plane_mask)[0],
            'nanoparticle': np.where(nanoparticle_mask)[0],
            'all': np.arange(len(data))
        }
        
    def get_atoms_spatial_coordinates(self, data):
        x_idx, y_idx, z_idx = self._atoms_spatial_coordinates_indices
        x = data[:, x_idx]
        y = data[:, y_idx]
        z = data[:, z_idx]
        return x, y, z
    
    def get_data(self, timestep_idx = None) -> np.ndarray:
        if not self._is_parsed:
            self._index_file()
        
        if timestep_idx is not None:
            if isinstance(timestep_idx, int):
                if timestep_idx < 0:
                    timestep_idx = len(self._timesteps) + timestep_idx
                return self._get_timestep_data(timestep_idx)
            else:
                # Handle timestep value instead of index
                for idx, timestep in enumerate(self._timesteps):
                    if timestep == timestep_idx:
                        return self._get_timestep_data(idx)
                raise ValueError(f"Timestep {timestep_idx} not found")
                
        # Load all data if requested (but warn about memory usage)
        if not self._data or len(self._data) < len(self._timesteps):
            print("WARNING: Loading all timesteps into memory. This may consume a lot of RAM.")
            self._load_all_timesteps()
            
        return self._data
    
    def get_headers(self) -> List[str]:
        return self._headers
    
    def get_timesteps(self) -> Dict[str, Any]:
        return self._timesteps
    
    def get_column_data(self, column_name: str, timestep_idx=-1, data=None) -> np.ndarray:
        try:
            column_idx = self._headers.index(column_name)
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in headers: {self._headers}")
        if data is not None:
            return data[:, column_idx]
        if timestep_idx < 0:
            timestep_idx = len(self._data) + timestep_idx
        return self._data[timestep_idx][:, column_idx]

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    def get_atom_types(self, timestep_idx=-1) -> np.ndarray:
        try:
            type_idx = self._headers.index('type')
        except ValueError:
            raise ValueError('Atom type information not found in headers')
        if timestep_idx < 0:
            timestep_idx = len(self._data) + timestep_idx
        return self._data[timestep_idx][:, type_idx].astype(int)

    def get_atom_count(self, timestep_idx=-1) -> int:
        if timestep_idx < 0:
            timestep_idx = len(self._data) + timestep_idx
        return len(self._data[timestep_idx])
    