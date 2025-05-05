from core.base_parser import BaseParser
import numpy as np

class CommonNeighborAnalysisAnalyzer:
    '''
    CNA values:
    0: Unknown/Disordered
    1: FCC (Face-centered cubic)
    2: HCP (Hexagonal close-packed)
    3: BCC (Body-centered cubic)
    4: ICO (Icosahedral)
    5: Other
    '''
    
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.structure_names = {
            0: 'Unknown/Disordered',
            1: 'FCC',
            2: 'BCC',
            3: 'BCC',
            4: 'ICO',
            5: 'Other'
        }
    
    def get_cna_data(self):
        data = self.parser.get_data()
        headers = self.parser.get_headers()
        cna_idx = headers.index('c_cna')
        cna_data = []
        for timestep_data in data:
            cna_values = timestep_data[:, cna_idx]
            cna_data.append(cna_values)
        return cna_data
    
    def get_structure_counts(self, timestep_idx=-1):
        data = self.parser.get_data()
        headers = self.parser.get_headers()
        if timestep_idx < 0:
            timestep_idx = len(data) + timestep_idx
        current_data = data[timestep_idx]
        cna_idx = headers.index('c_cna')
        cna_values = current_data[:, cna_idx]
        # Ocurrences of each structure type
        unique, counts = np.unique(cna_values, return_counts=True)
        structure_counts = dict(zip(unique, counts))
        return structure_counts
    
    def get_structure_evolution(self):
        data = self.parser.get_data()
        timesteps = self.parser.get_timesteps()
        evolution = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [] }
        for i in range(len(timesteps)):
            counts = self.get_structure_counts(i)
            total_atoms = sum(counts.values())
            # Fill in structure percentages, default to 0 if not present
            for structure_type in range(6):
                if structure_type in counts:
                    evolution[structure_type].append(counts[structure_type] / total_atoms * 100)
                else:
                    evolution[structure_type].append(0)
        return evolution
    
    def get_structure_percentages(self, timestep_idx=-1):
        counts = self.get_structure_counts(timestep_idx)
        total = sum(counts.values())
        percentages = {}
        for structure_type, count in counts.items():
            percentages[structure_type] = (count / total) * 100
        return percentages
    
    def get_spatial_distribution(self, timestep_idx=-1):
        data = self.parser.get_data()
        headers = self.parser.get_headers()
        
        if timestep_idx < 0:
            timestep_idx = len(data) + timestep_idx

        current_data = data[timestep_idx]
        x_idx = headers.index('x')
        y_idx = headers.index('y')
        z_idx = headers.index('z')
        cna_idx = headers.index('c_cna')

        x = current_data[:, x_idx]
        y = current_data[:, y_idx]
        z = current_data[:, z_idx]
        cna = current_data[:, cna_idx]

        return x, y, z, cna
    
    def compare_structures(self, timestep_idx1=0, timestep_idx2=-1):
        data = self.parser.get_data()
        
        if timestep_idx1 < 0:
            timestep_idx1 = len(data) + timestep_idx1
        if timestep_idx2 < 0:
            timestep_idx2 = len(data) + timestep_idx2
        
        structure_counts1 = self.get_structure_counts(timestep_idx1)
        structure_counts2 = self.get_structure_counts(timestep_idx2)

        total_atoms1 = sum(structure_counts1.values())
        total_atoms2 = sum(structure_counts2.values())

        # Combine keys 
        all_types = sorted(set(list(structure_counts1.keys()) + list(structure_counts2.keys())))
        comparison = {
            'types': all_types,
            'names': [self.structure_names.get(t, f"Type {t}") for t in all_types],
            'counts1': [structure_counts1.get(t, 0) for t in all_types],
            'counts2': [structure_counts2.get(t, 0) for t in all_types],
            'percentages1': [structure_counts1.get(t, 0)/total_atoms1*100 for t in all_types],
            'percentages2': [structure_counts2.get(t, 0)/total_atoms2*100 for t in all_types]
        }
        
        return comparison