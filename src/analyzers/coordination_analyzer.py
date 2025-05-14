from core.base_parser import BaseParser
import numpy as np
import cupy as cp

class CoordinationAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
    
    def get_coord_data(self, timestep_idx=-1):
        coord_values = self.parser.get_analysis_data('coord', timestep_idx)
        return coord_values
    
    def get_coord_distribution(self, timestep_idx=-1):
        coord_np = self.get_coord_data(timestep_idx)
        coord_gpu = cp.asarray(coord_np)
        unique_gpu, counts_gpu = cp.unique(coord_gpu, return_counts=True)
        total = coord_gpu.size
        percentages_gpu = counts_gpu / total * 100
        unique = cp.asnumpy(unique_gpu)
        counts = cp.asnumpy(counts_gpu)
        percentages = cp.asnumpy(percentages_gpu)
        return unique, counts, percentages
    
    def get_coord_stats(self, timestep_idx=-1):
        coord_np = self.get_coord_data(timestep_idx)
        coord_gpu = cp.asarray(coord_np)
        mean = float(cp.mean(coord_gpu).get())
        median = float(cp.median(coord_gpu).get())
        std = float(cp.std(coord_gpu).get())

        perfect_gpu = cp.sum(coord_gpu == 12)
        perfect_count = int(perfect_gpu.get())
        total = coord_gpu.size
        perfect_ratio = perfect_count / total * 100
        defect_ratio = 100 - perfect_ratio

        return {
            'mean': mean,
            'median': median,
            'std': std,
            'perfect_ratio': perfect_ratio,
            'defect_ratio': defect_ratio
        }
    
    def get_coord_evolution(self):
        timesteps = self.parser.get_timesteps()
        mean_list = []
        perfect_list = []
        defect_list = []
        for i in range(len(timesteps)):
            stats = self.get_coord_stats(i)
            mean_list.append(stats['mean'])
            perfect_list.append(stats['perfect_ratio'])
            defect_list.append(stats['defect_ratio'])
        return timesteps, mean_list, perfect_list, defect_list
    
    def get_spatial_distribution(self, timestep_idx=-1):
        data = self.parser.get_data(timestep_idx)
        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        coord = self.get_coord_data(timestep_idx)
        return x, y, z, coord
    
    def classify_atoms(self, timestep_idx=-1):
        coord_np = self.get_coord_data(timestep_idx)
        coord_gpu = cp.asarray(coord_np)
        perfect_gpu = cp.where(coord_gpu == 12)[0]
        surface_gpu = cp.where(coord_gpu < 9)[0]
        detect_gpu = cp.where((coord_gpu >= 9) & (coord_gpu < 12))[0]
        # Convert indices back to NumPy
        perfect = cp.asnumpy(perfect_gpu)
        surface = cp.asnumpy(surface_gpu)
        detect = cp.asnumpy(detect_gpu)
        return perfect, surface, detect
    
    def get_coord_range_distribution(self, timestep_idx=-1):
        coord_np = self.get_coord_data(timestep_idx)
        coord_gpu = cp.asarray(coord_np)
        total = coord_gpu.size

        # Define masks on GPU
        m1 = (coord_gpu >= 1) & (coord_gpu <= 4)
        m2 = (coord_gpu >= 5) & (coord_gpu <= 8)
        m3 = (coord_gpu >= 9) & (coord_gpu <= 11)
        m4 = coord_gpu == 12
        m5 = coord_gpu > 12

        counts_gpu = cp.stack([cp.sum(m1), cp.sum(m2), cp.sum(m3), cp.sum(m4), cp.sum(m5)])
        counts = cp.asnumpy(counts_gpu).tolist()
        percentages = [(c / total) * 100 for c in counts]

        ranges = ['1-4', '5-8', '9-11', '12 (perfect)', '13+']
        return ranges, counts, percentages


    def compare_timesteps(self, timestep_idx1=0, timestep_idx2=-1):
        timesteps = self.parser.get_timesteps()
        ranges, counts1, perc1 = self.get_coord_range_distribution(timestep_idx1)
        _, counts2, perc2 = self.get_coord_range_distribution(timestep_idx2)
        return {
            'timestep1': timesteps[timestep_idx1],
            'timestep2': timesteps[timestep_idx2],
            'ranges': ranges,
            'counts1': counts1,
            'counts2': counts2,
            'percentages1': perc1,
            'percentages2': perc2
        }