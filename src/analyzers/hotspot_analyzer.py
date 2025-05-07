from core.base_parser import BaseParser
import numpy as np

class HotspotAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser

    def get_hotspot_data(self, timestep_idx=-1):
        data = self.parser.get_data()
        headers = self.parser.get_headers()

        if timestep_idx < 0:
            timestep_idx = len(data) + timestep_idx
        
        current_data = data[timestep_idx]
        x, y, z = self.parser.get_atoms_spatial_coordinates(current_data)
        ke_idx = headers.index('c_ke_hotspots')
        hotspot_idx = headers.index('v_is_hotspot')

        ke_values = current_data[:, ke_idx]
        is_hotspot = current_data[:, hotspot_idx]

        return x, y, z, ke_values, is_hotspot
    
    def get_hotspot_stats(self, timestep_idx=-1):
        _, _, _, ke_values, is_hotspot = self.get_hotspot_data(timestep_idx)
        total_atoms = len(ke_values)
        hotspot_count = np.sum(is_hotspot > 0)
        hotspot_ratio = (hotspot_count / total_atoms) * 100 if total_atoms > 0 else 0
        average_energy = np.mean(ke_values)
        hotspot_mask = is_hotspot > 0
        if np.any(hotspot_mask):
            hotspot_average_energy = np.mean(ke_values[hotspot_mask])
        else:
            hotspot_average_energy = 0
        max_energy = np.max(ke_values)
        return {
            'total_atoms': total_atoms,
            'hotspot_count': hotspot_count,
            'hotspot_ratio': hotspot_ratio,
            'average_energy': average_energy,
            'hotspot_average_energy': hotspot_average_energy,
            'max_energy': max_energy
        }
    
    def get_energy_distribution(self, timestep_idx=-1, bins=50):
        _, _, _, ke_values, is_hotspot = self.get_hotspot_data(timestep_idx)
        histogram_values, energy_bins = np.histogram(ke_values, bins=bins)
        histogram_normalized = (histogram_values / len(ke_values)) * 100
        # Estimate threshold used for hotspot detection
        # in the LAMMPS script it was defined as a multiple of kB*T
        if np.any(is_hotspot  > 0):
            min_hotspot_energy = np.min(ke_values[is_hotspot > 0])
            threshold = min_hotspot_energy
        else:
            threshold = None
        return energy_bins, histogram_values, histogram_normalized, threshold

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