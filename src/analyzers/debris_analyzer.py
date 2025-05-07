from core.base_parser import BaseParser
import numpy as np

class DebrisAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser

    def get_cluster_data(self, timestep_idx=-1):
        data = self.parser.get_data()
        headers = self.parser.get_headers()

        if timestep_idx < 0:
            timestep_idx = len(data) + timestep_idx
        
        cluster_values = self.parser.get_column_data('c_cluster', timestep_idx)
        unique_clusters = np.unique(cluster_values)
        
        cluster_sizes = {}
        cluster_atoms = {}

        for cluster_id in unique_clusters:
            if cluster_id == 0: continue
            atom_indices = np.where(cluster_values == cluster_id)[0]
            cluster_sizes[cluster_id] = len(atom_indices)
            cluster_atoms[cluster_id] = atom_indices

        return unique_clusters, cluster_sizes, cluster_atoms
    
    def get_cluster_evolution(self):
        timesteps = self.parser.get_timesteps()
        num_clusters = []
        largest_cluster = []
        average_cluster = []

        for i in range(len(timesteps)):
            unique_cluster, cluster_sizes, _ = self.get_cluster_data(i)
            valid_clusters = [cluster_id for cluster_id in unique_cluster if cluster_id != 0]
            if len(valid_clusters) > 0:
                num_clusters.append(len(valid_clusters))
                sizes = [cluster_sizes[cluster_id] for cluster_id in valid_clusters]
                largest_cluster.append(max(sizes) if sizes else 0)
                average_cluster.append(np.mean(sizes) if sizes else 0)
                continue
            num_clusters.append(0)
            largest_cluster.append(0)
            average_cluster.append(0)

        return timesteps, num_clusters, largest_cluster, average_cluster

    def get_cluster_size_distribution(self, timestep_idx=-1, min_size=2):
        _, cluster_sizes, _ = self.get_cluster_data(timestep_idx)
        filtered_sizes = { key: value for key, value in cluster_sizes.items() if value >= min_size }
        size_distribution = {}
        for size in filtered_sizes.values():
            if size in size_distribution:
                size_distribution[size] += 1
            else:
                size_distribution[size] = 1
        sizes = sorted(size_distribution.keys())
        counts = [size_distribution[size] for size in sizes]
        return sizes, counts
    
    def get_cluster_spatial_data(self, timestep_idx=-1, min_size=2):
        data = self.parser.get_data()

        if timestep_idx < 0:
            timestep_idx = len(data) + timestep_idx
        
        current_data = data[timestep_idx]
        x, y, z = self.parser.get_atoms_spatial_coordinates(current_data)

        cluster_values = self.parser.get_column_data('c_cluster', timestep_idx)

        _, cluster_sizes, cluster_atoms = self.get_cluster_data(timestep_idx)
        filtered_clusters = { key: value for key, value in cluster_sizes.items() if value >= min_size }
        cluster_positions = {}
        for cluster_id, atom_indices in cluster_atoms.items():
            if cluster_id in filtered_clusters:
                cluster_positions[cluster_id] = (
                    np.mean(x[atom_indices]),
                    np.mean(y[atom_indices]),
                    np.mean(z[atom_indices])
                )
        atom_coords = {
            'x': x,
            'y': y,
            'z': z,
            'cluster_id': cluster_values
        }
        return cluster_positions, filtered_clusters, atom_coords

    def get_largest_clusters(self, timestep_idx=-1, n=5):
        _, clusters_sizes, _ = self.get_cluster_data(timestep_idx)
        sorted_clusters = sorted(clusters_sizes.items(), key=lambda x: x[1], reverse=True)
        top_clusters = sorted_clusters[:n]
        return top_clusters