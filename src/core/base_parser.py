from abc import ABC, abstractmethod
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
        self._atoms_spatial_coordinates_indices.append(x_idx, y_idx, z_idx)

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
    
    def get_atoms_spatial_coordinates(self, data = None):
        if not self._is_parsed:
            self.parse()
        if data is None:
            data = self._data
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