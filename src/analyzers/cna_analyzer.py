from core.base_parser import BaseParser
import cupy as cp
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
            2: 'HCP',
            3: 'BCC',
            4: 'ICO',
            5: 'Other'
        }
    
    def get_cna_data(self):
        cna_data = []
        for i in range(len(self.parser.get_timesteps())):
            cna_data.append(self.parser.get_analysis_data('cna', i))
        return cna_data
    
    def get_structure_counts(self, timestep_idx=-1):
        cna_values = self.parser.get_analysis_data('cna', timestep_idx)
        cna_gpu = cp.asarray(cna_values, dtype=cp.int32)
        counts_gpu = cp.bincount(cna_gpu, minlength=6)
        counts = cp.asnumpy(counts_gpu)
        return { i: int(counts[i]) for i in range(counts.shape[0]) }
    
    def get_structure_evolution(self):
        timesteps = self.parser.get_timesteps()
        all_counts = []
        for i in range(len(timesteps)):
            cna_np = self.parser.get_analysis_data('cna', i)
            cna_gpu = cp.asarray(cna_np, dtype=cp.int32)
            all_counts.append(cp.bincount(cna_gpu, minlength=6))
        counts_matrix = cp.stack(all_counts)
        totals = counts_matrix.sum(axis=1, keepdims=True)
        pct_gpu = counts_matrix / totals * 100
        pct = cp.asnumpy(pct_gpu)
        return { t: pct[:, t].tolist() for t in range(6) }
    
    def get_structure_percentages(self, timestep_idx=-1):
        counts = self.get_structure_counts(timestep_idx)
        total = sum(counts.values())
        percentages = {}
        for structure_type, count in counts.items():
            percentages[structure_type] = (count / total) * 100
        return percentages
    
    def get_spatial_distribution(self, timestep_idx=-1):        
        data = self.parser.get_data(timestep_idx)
        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        cna = self.parser.get_analysis_data('cna', timestep_idx)

        return x, y, z, cna
    
    def compare_structures(self, timestep_idx1=0, timestep_idx2=-1):
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