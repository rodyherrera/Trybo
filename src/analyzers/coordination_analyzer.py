from core.base_parser import BaseParser
import numpy as np

class CoordinationAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
    
    def get_coord_data(self, timestep_idx=-1):
        coord_values = self.parser.get_analysis_data('coord', timestep_idx)
        return coord_values
    
    def get_coord_distribution(self, timestep_idx=-1):
        coord_data = self.get_coord_data(timestep_idx)
        unique, counts = np.unique(coord_data, return_counts=True)
        total = len(coord_data)
        percentages = (counts / total) * 100
        return unique, counts, percentages
    
    def get_coord_stats(self, timestep_idx=-1):
        coord_data = self.get_coord_data(timestep_idx)
        mean = np.mean(coord_data)
        median = np.median(coord_data)
        std = np.std(coord_data)

        # FCC coordination is 12 (We're using Copper in .lammps)
        # TODO: this needs to be dynamic.
        perfect_count = np.sum(coord_data == 12)
        perfect_ratio = (perfect_count / len(coord_data)) * 100
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
        mean_coord = []
        perfect_ratio = []
        defect_ratio = []
        for i in range(len(timesteps)):
            stats = self.get_coord_stats(i)
            mean_coord.append(stats['mean'])
            perfect_ratio.append(stats['perfect_ratio'])
            defect_ratio.append(stats['defect_ratio'])
        return timesteps, mean_coord, perfect_ratio, defect_ratio
    
    def get_spatial_distribution(self, timestep_idx=-1):
        data = self.parser.get_data(timestep_idx)
        x, y, z = self.parser.get_atoms_spatial_coordinates(data)
        coord = self.parser.get_analysis_data('coord', timestep_idx)
        return x, y, z, coord
    
    def classify_atoms(self, timestep_idx=-1):
        coord_data = self.get_coord_data(timestep_idx)
        perfect = np.where(coord_data == 12)[0]
        surface = np.where(coord_data < 9)[0]
        detect = np.where((coord_data >= 9) & (coord_data < 12))[0]
        return perfect, surface, detect
    
    def get_coord_range_distribution(self, timestep_idx=-1):
        coord_data = self.get_coord_data(timestep_idx)
        ranges = ['1-4', '5-8', '9-11', '12 (perfect)', '13+']
        count_1_4 = np.sum((coord_data >= 1) & (coord_data <= 4))
        count_5_8 = np.sum((coord_data >= 5) & (coord_data <= 8))
        count_9_11 = np.sum((coord_data >= 9) & (coord_data <= 11))
        count_12 = np.sum(coord_data == 12)
        count_13_plus = np.sum(coord_data > 12)
        counts = [count_1_4, count_5_8, count_9_11, count_12, count_13_plus]
        total = len(coord_data)
        percentages = [(count / total) * 100 for count in counts]
        return ranges, counts, percentages

    def compare_timesteps(self, timestep_idx1=0, timestep_idx2=-1):
        data = self.parser.get_data()
        timesteps = self.parser.get_timesteps()
        ranges1, counts1, percentages1 = self.get_coord_range_distribution(timestep_idx1)
        ranges2, counts2, percentages2 = self.get_coord_range_distribution(timestep_idx2)
        return {
            'timestep1': timesteps[timestep_idx1],
            'timestep2': timesteps[timestep_idx2],
            # NOTE: ranges1 should be == ranges2!
            'ranges': ranges1,
            'counts1': counts1,
            'counts2': counts2,
            'percentages1': percentages1,
            'percentages2': percentages2
        }