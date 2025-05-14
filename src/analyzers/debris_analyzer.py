from core.base_parser import BaseParser
import numpy as np
import cupy as cp

class DebrisAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser

    def get_cluster_data(self, timestep_idx=-1):
        cluster_np = self.parser.get_analysis_data('cluster', timestep_idx)
        cluster_gpu = cp.asarray(cluster_np, dtype=cp.int32)
        unique_gpu = cp.unique(cluster_gpu)
        unique = cp.asnumpy(unique_gpu)
        cluster_sizes = {}
        cluster_atoms = {}
        for cluster_id in unique:
            if cluster_id == 0: continue
            idx_gpu = cp.where(cluster_gpu == cluster_id)[0]
            idx = cp.asnumpy(idx_gpu)
            cluster_sizes[int(cluster_id)] = idx.size
            cluster_atoms[int(cluster_id)] = idx
        return unique, cluster_sizes, cluster_atoms
    
    def get_cluster_evolution(self):
        timesteps = self.parser.get_timesteps()
        num_clusters = []
        largest_cluster = []
        average_cluster = []

        for i in range(len(timesteps)):
            _, sizes, _ = self.get_cluster_data(i)
            values = list(sizes.values())
            if values:
                num_clusters.append(len(values))
                largest_cluster.append(max(values))
                average_cluster.append(float(np.mean(values)))
            else:
                num_clusters.append(0)
                largest_cluster.append(0)
                average_cluster.append(0.0)

        return timesteps, num_clusters, largest_cluster, average_cluster

    def get_cluster_size_distribution(self, timestep_idx=-1, min_size=2):
        _, sizes, _ = self.get_cluster_data(timestep_idx)
        filtered = [s for s in sizes.values() if s >= min_size]
        if not filtered:
            return [], []
        sizes_gpu = cp.asarray(filtered, dtype=cp.int32)
        unique_gpu, counts_gpu = cp.unique(sizes_gpu, return_counts=True)
        unique = cp.asnumpy(unique_gpu)
        counts = cp.asnumpy(counts_gpu)
        order = np.argsort(unique)
        return unique[order].tolist(), counts[order].tolist()
    
    def get_cluster_spatial_data(self, timestep_idx=-1, min_size=2):
        data = self.parser.get_data(timestep_idx)
        x_np, y_np, z_np = self.parser.get_atoms_spatial_coordinates(data)
        x_gpu, y_gpu, z_gpu = cp.asarray(x_np), cp.asarray(y_np), cp.asarray(z_np)

        _, sizes, atoms = self.get_cluster_data(timestep_idx)
        positions = {}

        for cid, size in sizes.items():
            if size >= min_size:
                idx = atoms[cid]
                idx_gpu = cp.asarray(idx, dtype=cp.int32)
                mean_x = float(cp.mean(x_gpu[idx_gpu]).get())
                mean_y = float(cp.mean(y_gpu[idx_gpu]).get())
                mean_z = float(cp.mean(z_gpu[idx_gpu]).get())
                positions[cid] = (mean_x, mean_y, mean_z)

        atom_coords = {
            'x': x_np,
            'y': y_np,
            'z': z_np,
            'cluster_id': self.parser.get_analysis_data('cluster', timestep_idx)
        }
        return positions, {cid: sizes[cid] for cid in positions}, atom_coords
    
    def get_largest_clusters(self, timestep_idx: int = -1, n: int = 5):
        _, sizes, _ = self.get_cluster_data(timestep_idx)
        sorted_clusters = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_clusters[:n]