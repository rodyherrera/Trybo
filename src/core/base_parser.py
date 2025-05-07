from abc import ABC
from typing import List, Dict, Any
import numpy as np

class BaseParser(ABC):
    def __init__(self, filename: str):
        self.filename = filename

        self._timesteps = []
        self._data = []
        self._headers = []
        self._is_parsed = False

        # The x, y, and z values ​​in the LAMMPS dump files.
        # ITEM: ATOMS id type x y z [...]
        self._atoms_spatial_coordinates_indices = []

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

    def parse(self):
        timesteps = []
        data = []
        headers = []

        with open(self.filename, 'r') as file:
            line = file.readline()
            while line:
                if 'ITEM: TIMESTEP' in line:
                    timestep = int(file.readline().strip())
                    timesteps.append(timestep)
                    
                    while line and 'ITEM: ATOMS' not in line:
                        line = file.readline()
                        if not line:
                            break
                    
                    if line and 'ITEM: ATOMS' in line:
                        headers = line.split()[2:]
                        
                        atoms_data = []
                        line = file.readline()
                        while line and 'ITEM:' not in line:
                            values = [float(val) for val in line.split()]
                            atoms_data.append(values)
                            line = file.readline()
                        
                        data.append(np.array(atoms_data))
                        continue
                
                line = file.readline()
    
        if headers is None:
            raise ValueError('No headers were found in the file. Check the format.')

        self._timesteps = timesteps
        self._data = data
        self._headers = headers

        self._parse_atoms_spatial_coordinates_indices()

        self._is_parsed = True

        return timesteps, data, headers
    
    def get_atom_group_indices(self, data=None, timestep_idx=-1):
        '''
        Identifies atom indices for different groups based on their position along the Z axis
        
        Args:
            data: Optional data array to use instead of the parser's data
            timestep_idx: Timestep index to use if data is not provided

        Returns:
            dict: Dictionary with indices for each group (lower_plane, upper_plane, nanoparticle, all)
        '''
        if data is None:
            data = self.get_data()[timestep_idx]
        # Get Z coordinates
        x, y, z = self.get_atoms_spatial_coordinates(timestep_idx=timestep_idx)
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
            'all': np.arange(len(z))
        }
        
    def get_atoms_spatial_coordinates(self, data):
        if not self._is_parsed:
            self.parse()
        x_idx, y_idx, z_idx = self._atoms_spatial_coordinates_indices
        x = data[:, x_idx]
        y = data[:, y_idx]
        z = data[:, z_idx]
        return x, y, z
    
    def get_data(self) -> np.ndarray:
        if not self._is_parsed:
            self.parse()
        return self._data
    
    def get_headers(self) -> List[str]:
        if not self._is_parsed:
            self.parse()
        return self._headers
    
    def get_timesteps(self) -> Dict[str, Any]:
        if not self._is_parsed:
            self.parse()
        return self._timesteps