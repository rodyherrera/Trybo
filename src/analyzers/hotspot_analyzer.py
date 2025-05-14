from core.base_parser import BaseParser
import numpy as np
import cupy as cp

class HotspotAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser

    def get_hotspot_data(self, timestep_idx=-1):
        data = self.parser.get_data(timestep_idx)
        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        ke_values = self.parser.get_analysis_data('ke_hotspots', timestep_idx)
        is_hotspot = self.parser.get_analysis_data('is_hotspot', timestep_idx)
        return x, y, z, ke_values, is_hotspot
    
    def get_hotspot_stats(self, timestep_idx=-1):
        _, _, _, ke_np, mask_np = self.get_hotspot_data(timestep_idx)
        ke_gpu = cp.asarray(ke_np, dtype=cp.float64)
        mask_gpu = cp.asarray(mask_np, dtype=cp.bool_)
        total = ke_gpu.size
        hotspot_cnt = int(cp.sum(mask_gpu).get())
        hotspot_pct = hotspot_cnt / total * 100 if total else 0.8
        average_energy = float(cp.mean(ke_gpu).get())
        max_energy = float(cp.max(ke_gpu).get())
        hotspot_average = float(cp.mean(ke_gpu[mask_gpu]).get()) if hotspot_cnt else 0.0
        return {
            'total_atoms': total,
            'hotspot_count': hotspot_cnt,
            'hotspot_ratio': hotspot_pct,
            'average_energy': average_energy,
            'hotspot_average_energy': hotspot_average,
            'max_energy': max_energy
        }
    
    def get_energy_distribution(self, timestep_idx=-1, bins=50):
        _, _, _, ke_np, mask_np = self.get_hotspot_data(timestep_idx)
        ke_gpu = cp.asarray(ke_np, dtype=cp.float64)
        histogram_gpu, bin_edges_gpu = cp.histogram(ke_gpu, bins=bins)
        histogram = cp.asnumpy(histogram_gpu)
        bin_edges = cp.asnumpy(bin_edges_gpu)
        norm_pct = histogram / histogram.sum() * 100 if histogram.sum() else histogram
        # threshold used in LAMMPS, minimum hotspot energy
        mask_gpu = cp.asarray(mask_np, dtype=cp.bool_)
        if cp.any(mask_gpu):
            threshold = float(cp.min(ke_gpu[mask_gpu]).get())
        else:
            threshold = None
        return bin_edges, histogram, norm_pct, threshold

    def get_hotspot_evolution(self):
        timesteps = self.parser.get_timesteps()
        hotspot_counts = []
        hotspot_ratios = []
        average_eneries = []
        max_energies = []
        for i in range(len(timesteps)):
            stats = self.get_hotspot_stats(i)
            hotspot_counts.append(stats['hotspot_count'])
            hotspot_ratios.append(stats['hotspot_ratio'])
            average_eneries.append(stats['average_energy'])
            max_energies.append(stats['max_energy'])
        return timesteps, hotspot_counts, hotspot_ratios, average_eneries, max_energies
        
    def get_hotspot_clusters(self, timestep_idx=-1, cutoff=3.0):
        x, y, z, _, is_hotspot = self.get_hotspot_data(timestep_idx)
        hotspot_atoms_indices = np.where(is_hotspot > 0)[0]
        if len(hotspot_atoms_indices) == 0:
            return {}, {}, {}
        hotspot_atoms_x = x[hotspot_atoms_indices]
        hotspot_atoms_y = y[hotspot_atoms_indices]
        hotspot_atoms_z = z[hotspot_atoms_indices]
        clusters = {}
        cluster_id = 0
        unassigned = set(range(len(hotspot_atoms_indices)))
        while unassigned:
            current_cluster = []
            seed = next(iter(unassigned))
            queue = [seed]
            unassigned.remove(seed)
            while queue:
                current = queue.pop(0)
                current_cluster.append(current)
                # Find neighbors
                for i in list(unassigned):
                    dx = hotspot_atoms_x[current] - hotspot_atoms_x[i]
                    dy = hotspot_atoms_y[current] - hotspot_atoms_y[i]
                    dz = hotspot_atoms_z[current] - hotspot_atoms_z[i]
                    distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if distance <= cutoff:
                        queue.append(i)
                        unassigned.remove(i)
            clusters[cluster_id] = hotspot_atoms_indices[np.array(current_cluster)]
            cluster_id += 1
        cluster_sizes = { cluster_id: len(indices) for cluster_id, indices in clusters.items() }
        cluster_positions = {}
        for cluster_id, indices in clusters.items():
            cluster_positions[cluster_id] = (np.mean(x[indices]), np.mean(y[indices]), np.mean(z[indices]))
        return clusters, cluster_sizes, cluster_positions

    def get_spatial_energy_distribution(self, timestep_idx=-1):
        return self.get_hotspot_data(timestep_idx)