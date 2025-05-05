import numpy as np
from typing import List, Tuple
from core.base_parser import BaseParser

class CenterSymmetryParser(BaseParser):
    def __init__(self, filename: str, threshold: float = 8.0):
        super().__init__(filename)
        self.threshold = threshold
        self.property_name = 'Centro-Symmetry Parameter'
        self._data = None
        self._box_size = None
        self._column_names = None
    
    def parse(self) -> Tuple[np.ndarray, List, List[str]]:
        print(f'Reading file: {self.filename}')
        with open(self.filename, 'r') as file:
            lines = file.readlines()
        
        # Find the number of atoms
        total_atoms = int(lines[3])
        print(f'Total atoms: {total_atoms}')

        # Find the box dimensions
        box_size_x = [float(x) for x in lines[5].split()]
        box_size_y = [float(x) for x in lines[6].split()]
        box_size_z = [float(x) for x in lines[7].split()]
        
        # Find column names from ITEM: ATOMS line
        header_line = lines[8].strip()
        if header_line.startswith('ITEM: ATOMS'):
            column_names = header_line.replace('ITEM: ATOMS', '').strip().split()
            print(f'Detected columns: {column_names}')
        else:
            column_names = []
            print('Warning: Could not detect column names from dump file')
        
        # Find where the atoms data begins
        start_line = 9
        
        # Read atom data
        atom_data = []
        for i in range(start_line, start_line + total_atoms):
            if i >= len(lines):
                break

            values = lines[i].split()
            
            # Need at least id, type, x, y, z, centro
            if len(values) >= 6:
                try:
                    atom_id = int(values[0])
                    atom_type = int(values[1])
                    x = float(values[2])
                    y = float(values[3])
                    z = float(values[4])
                    centro = float(values[5])
                    atom_data.append([atom_id, atom_type, x, y, z, centro])
                except (ValueError, IndexError) as e:
                    print(f'Error parsing line {i}: {lines[i]} - {e}')

        # Convert to numpy array for efficient operations
        atom_data_array = np.array(atom_data)
        box_size = [box_size_x, box_size_y, box_size_z]
        
        # Store parsed data
        self._data = atom_data_array
        self._box_size = box_size
        self._column_names = column_names
        
        return atom_data_array, box_size, column_names
    
    def get_data(self) -> np.ndarray:
        if self._data is None:
            self._data, self._box_size, self._column_names = self.parse()
        return self._data
    
    def get_box_size(self) -> List:
        if self._box_size is None:
            self._data, self._box_size, self._column_names = self.parse()
        return self._box_size
    
    def get_column_names(self) -> List[str]:
        if self._column_names is None:
            self._data, self._box_size, self._column_names = self.parse()
        return self._column_names
    
    def get_property_column_index(self) -> int:
        column_names = self.get_column_names()
        
        # Try to find the centro-symmetry column
        for i, col in enumerate(column_names):
            if 'c_center_symmetric' in col or 'centro' in col.lower():
                return i
                
        # If not found, assume it's the last column after x,y,z
         # Fallback to default position (6th column in 0-index)
        return 5
    
    def get_defect_data(self) -> Tuple[np.ndarray, float]:
        '''Get data about defects based on threshold'''
        data = self.get_data()
        prop_idx = self.get_property_column_index()
        
        if prop_idx >= data.shape[1]:
            print(f'Warning: Property column index {prop_idx} out of bounds')
            return np.array([]), 0.0
            
        values = data[:, prop_idx]
        defects = data[values > self.threshold]
        defect_percentage = (len(defects) / len(data)) * 100 if len(data) > 0 else 0
        
        return defects, defect_percentage