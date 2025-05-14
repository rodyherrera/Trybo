from core.base_parser import BaseParser
from utilities.analyzer import get_data_from_coord_axis
import numpy as np
import cupy as cp

class EnergyAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None

    def get_energy_statistics(self, timestep_idx=-1, group=None, energy_type='total'):
        # Retrieve raw energy values and move to GPU
        energy_column = self.get_energy_column_by_type(energy_type)
        energy_cpu = self.parser.get_analysis_data(energy_column, timestep_idx)
        energy_gpu = cp.asarray(energy_cpu, dtype=cp.float64)
        if group and group != 'all':
            indices = self.parser.get_atom_group_indices(self.parser, timestep_idx)[group]
            energy_gpu = energy_gpu[indices]
        mean = float(cp.mean(energy_gpu).get())
        median = float(cp.median(energy_gpu).get())
        max = float(cp.max(energy_gpu).get())
        min = float(cp.min(energy_gpu).get())
        std = float(cp.std(energy_gpu).get())
        sum = float(cp.sum(energy_gpu).get())
        return {
            'mean': mean,
            'median': median,
            'max': max,
            'min': min,
            'std': std,
            'sum': sum
        }

    def get_energy_evolution(self, group=None, energy_type='total'):
        timesteps = self.parser.get_timesteps()
        average_energy = []
        max_energy = []
        min_energy = []
        sum_energy = []
        for idx in range(len(timesteps)):
            stats = self.get_energy_statistics(idx, group, energy_type)
            average_energy.append(stats['mean'])
            max_energy.append(stats['max'])
            min_energy.append(stats['min'])
            sum_energy.append(stats['sum'])
        return timesteps, average_energy, max_energy, min_energy, sum_energy
    
    def get_high_energy_regions(self, timestep_idx=-1, threshold_percentile=95, energy_type='total', group=None):
        # Fetch per-atom energy and optionally filter by group
        data = self.parser.get_data(timestep_idx)
        if group and group != 'all':
            indices = self.parser.get_atom_group_indices(self.parser, timestep_idx)[group]
            data = data[indices]
        energy_column = self.get_energy_column_by_type(energy_type)
        energy_cpu = data[:, energy_column]
        energy_gpu = cp.asarray(energy_cpu, dtype=cp.float64)
        if energy_type == 'potential':
            # Most negative values = most stable
            threshold = cp.percentile(energy_gpu, 100 - threshold_percentile)
            mask_gpu = energy_gpu <= threshold
        else:
            threshold = cp.percentile(energy_gpu, threshold_percentile)
            mask_gpu = energy_gpu >= threshold
        # Gather high-energy atoms
        mask = cp.asnumpy(mask_gpu)
        high_energy_data = data[mask]
        return high_energy_data, mask
    
    def get_energy_column_by_type(self, energy_type):
        types = {
            # c_ke_mobile
            'kinetic': 'ke_mobile',
            # c_pe_mobile
            'potential': 'pe_mobile',
            # v_total_energy
            'total': 'total_energy'
        }
        return types.get(energy_type, 'total_energy')

    def calculate_energy_profile(self, timestep_idx=-1, axis='z', n_bins=20, energy_type='total'):
        data = self.parser.get_data(timestep_idx)
        coords_cpu = get_data_from_coord_axis(axis, self.parser.get_atoms_spatial_coordinates(data))
        energy_column = self.get_energy_column_by_type(energy_type)
        energy_cpu = data[:, energy_column]
        coords_gpu = cp.asarray(coords_cpu, dtype=cp.float64)
        energy_gpu = cp.asarray(energy_cpu, dtype=cp.float64)
        bin_edges_gpu = cp.linspace(coords_gpu.min(), coords_gpu.max(), n_bins + 1)
        bin_centers_cpu = 0.5 * (cp.asnumpy(bin_edges_gpu[:-1]) + cp.asnumpy(bin_edges_gpu[1:]))
        bin_indices = cp.digitize(coords_gpu, bin_edges_gpu)
        profile_gpu = cp.zeros(n_bins, dtype=cp.float64)
        for bin_id in range(1, n_bins + 1):
            mask = bin_indices == bin_id
            if cp.any(mask):
                profile_gpu[bin_id - 1] = energy_gpu[mask].mean()
        profile_cpu = cp.asnumpy(profile_gpu)
        return bin_centers_cpu.tolist(), profile_cpu.tolist()