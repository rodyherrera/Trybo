from typing import List, Tuple
from core.base_parser import BaseParser
import numpy as np

class EnergyParser(BaseParser):
    def __init__(self, filename: str):
        super().__init__(filename)
        self._data = None
        self._column_names = None

    def parse(self) -> Tuple[np.ndarray, List, List[str]]:
        print(f'Reading file: {self.filename}')
        timesteps = []
        all_data = []
        headers = None 

        with open(self.filename, 'r') as file:
            line = file.readline()
            while line:
                if 'ITEM: TIMESTEP' in line:
                    # Obtener el timestep
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
                        
                        all_data.append(np.array(atoms_data))
                        continue
                
                line = file.readline()

        if headers is None:
            raise ValueError('No headers were found in the file. Check the format.')

        return timesteps, all_data, headers