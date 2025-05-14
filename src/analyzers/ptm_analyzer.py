from core.base_parser import BaseParser
import numpy as np
import cupy as cp

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
        # Load the per-atom PTM type and optionally filter by atom group
        ptm_columns = self.parser.get_analysis_data('ptm', timestep_idx)
        ptm_types_cpu = ptm_columns[0].astype(int)
        ptm_types_gpu = cp.asarray(ptm_types_cpu, dtype=cp.int32)
        if group and group != 'all':
            positions = self.parser.get_data(timestep_index)
            group_indices = self.parser.get_atom_group_indices(positions)[group]
            ptm_types_gpu = ptm_types_gpu[group_indices]

        # Count each structure type on GPU
        max_type = max(self.structure_labels.keys())
        counts_gpu = cp.bincount(ptm_types_gpu, minlength=max_type + 1)
        counts_cpu = cp.asnumpy(counts_gpu)

        # Map to labels
        distribution = {
            self.structure_labels[stype]: int(counts_cpu[stype])
            for stype in self.structure_labels
        }
        return distribution
    
    def get_rmsd_statistics(self, timestep_idx=-1, group=None):
        ptm_columns = self.parser.get_analysis_data('ptm', timestep_idx)
        rmsd_cpu = ptm_columns[1]
        rmsd_gpu = cp.asarray(rmsd_cpu, dtype=cp.float64)
        if group and group != 'all':
            positions = self.parser.get_data(timestep_idx)
            group_indices = self.parser.get_atom_group_indices(positions)[group]
            rmsd_gpu = rmsd_gpu[group_indices]
        valid_mask = cp.isfinite(rmsd_gpu)
        valid_rmsd_gpu = rmsd_gpu[valid_mask]
        if valid_rmsd_gpu.size > 0:
            mean = float(cp.mean(valid_rmsd_gpu).get())
            median = float(cp.median(valid_rmsd_gpu).get())
            max = float(cp.max(valid_rmsd_gpu).get())
            min = float(cp.min(valid_rmsd_gpu).get())
            std = float(cp.std(valid_rmsd_gpu).get())
        else:
            mean = median = max = min = std = float('nan')

        return {
            'mean_rmsd': mean,
            'median_rmsd': median,
            'max_rmsd': max,
            'min_rmsd': min,
            'std_rmsd': std
        }
    
    def get_structure_evolution(self, group=None):
        timesteps = self.parser.get_timesteps()
        evolution = { name: [] for name in self.structure_names.values() }
        for idx in range(len(timesteps)):
            counts = self.get_structure_distribution(idx, group)
            for name, count in counts.items():
                evolution[name].append(count)
        return timesteps, evolution