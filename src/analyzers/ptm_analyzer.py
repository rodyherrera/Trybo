import numpy as np
from core.base_parser import BaseParser
from utilities.analyzer import get_atom_group_indices

class PTMAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser

        self.structure_names = {
            0: 'Other',
            1: 'FCC',
            2: 'HCP',
            3: 'BCC',
            4: 'ICO',
            5: 'SC'
        }

        self.structure_colors = {
            # Other/Unknown
            0: 'gray',
            # FCC
            1: 'green',
            # HCP
            2: 'blue',
            # BCC
            3: 'red',
            # ICO
            4: 'purple',
            # SC
            5: 'orange'
        }

        self._atoms_groups = None
    
    def get_structure_distribution(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        ptm_data = self.parser.get_analysis_data('ptm', timestep_idx)
        structure_types = ptm_data[0].astype(int)

        data = self.parser.get_data(timestep_idx)
        if group is not None and group != 'all':
            group_indices = self.parser.get_atom_group_indices(data)[group]
            structure_types = structure_types[group_indices]
        
        counts = {}
        for struct_type, name in self.structure_names.items():
            counts[name] = np.sum(structure_types == struct_type)

        return counts
    
    def get_rmsd_statistics(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()
        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        ptm_data = self.parser.get_analysis_data('ptm', timestep_idx)
        rmsd_values = ptm_data[1]

        data = self.parser.get_data(timestep_idx)
        if group is not None and group != 'all':
            group_indices = self.parser.get_atom_group_indices(data)[group]
            rmsd_values = rmsd_values[group_indices]
        valid_rmsd = rmsd_values[~np.isinf(rmsd_values) & ~np.isnan(rmsd_values)]
    
        if len(valid_rmsd) > 0:
            stats = {
                'mean': np.mean(valid_rmsd),
                'median': np.median(valid_rmsd),
                'max': np.max(valid_rmsd),
                'min': np.min(valid_rmsd),
                'std': np.std(valid_rmsd)
            }
        else:
            stats = {
                'mean': float('nan'),
                'median': float('nan'),
                'max': float('nan'),
                'min': float('nan'),
                'std': float('nan')
            }
        return stats

    def get_structure_evolution(self, group=None):
        timesteps = self.parser.get_timesteps()
        evolution = { name: [] for name in self.structure_names.values() }
        for idx in range(len(timesteps)):
            counts = self.get_structure_distribution(idx, group)
            for name, count in counts.items():
                evolution[name].append(count)
        return timesteps, evolution