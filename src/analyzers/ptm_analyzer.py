import numpy as np

class PTMAnalyzer:
    def __init__(self, parser):
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

    def get_atom_group_indices(self):
        if self._atoms_groups is not None:
            return self._atoms_groups
        data = self.parser.get_data()[0]
        x, y, z = self.parser.get_atoms_spatial_coordinates(data)

        # Dimensions
        z_min = np.min(z)
        z_max = np.max(z)
        z_threshold_lower = z_min + 2.5
        z_threshold_upper = z_max - 2.5

        # Indexes for each group
        lower_plane_mask = z <= z_threshold_lower
        upper_plane_mask = z >= z_threshold_upper
        nanoparticle_mask = ~(lower_plane_mask | upper_plane_mask)

        self._atoms_groups = {
            'lower_plane': np.where(lower_plane_mask)[0],
            'upper_plane': np.where(upper_plane_mask)[0],
            'nanoparticle': np.where(nanoparticle_mask)[0],
            'all': np.arange(len(data))
        }

        return self._atoms_groups
    
    def get_structure_distribution(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_timesteps()

        if timestep_idx < 0:
            timestep_idx = len(timesteps) + timestep_idx
        
        data = self.parser.get_data()[timestep_idx]

        if group is not None and group != 'all':
            group_indices = self.get_atom_group_indices()[group]
            data = data[group_indices]
        
        structure_types = data[:, 5].astype(int)

        counts = {}
        for struct_type, name in self.structure_names.items():
            counts[name] = np.sum(structure_types == struct_type)

        return counts
    
    def get_rmsd_statistics(self, timestep_idx=-1, group=None):
        timesteps = self.parser.get_data()[timestep_idx]
        
        if group is not None and group != 'all':
            group_indices = self.get_atom_group_indices()[group]
            data = data[group_indices]
        
        rmsd_values = data[:, 6]
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