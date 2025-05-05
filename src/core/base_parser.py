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
        self._is_parsed = True

        return timesteps, data, headers
    
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