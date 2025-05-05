from typing import List, Tuple
from core.base_parser import BaseParser
import numpy as np

class DumpParser(BaseParser):
    def parse(self) -> Tuple[np.ndarray, List, List[str]]:
        print(f'Reading file: {self.filename}')
        with open(self.filename, 'r') as file:
            lines = file.readlines()
        
        total_atoms = int(lines[3])
        print(f'Total atoms: {total_atoms}')

        box_size_x = [float(x) for x in lines[5].split()]
        box_size_y = [float(x) for x in lines[6].split()]
        box_size_z = [float(x) for x in lines[7].split()]
        box_size = [box_size_x, box_size_y, box_size_z]

        # Find column names from ITEM: ATOMS line
        header_line = lines[8].strip()
        if header_line.startswith('ITEM: ATOMS'):
            column_names = header_line.replace('ITEM: ATOMS', '').strip().split()
            print(f'Detected columns: {column_names}')
        else:
            column_names = []
            print('Warning: Could not detect column names from dump file')
        
        # Find where the atoms data begin
        start_line = 9
        atom_data = []
        for i in range(start_line, start_line + total_atoms):
            if i >= len(lines):
                break

            values = lines[i].split()
            
            # Need at least id, type, x, y, z
            if len(values) >= 5:
                try:
                    row = [float(val) for val in values[:len(column_names)]]
                    atom_data.append(row)
                except (ValueError, IndexError) as e:
                    print(f'Error parsing line {i}: {lines[i]} - {e}')
                    
        self.metadata = {
            'total_atoms': total_atoms,
            'filename': self.filename
        }
        
        return np.array(atom_data), box_size, column_names