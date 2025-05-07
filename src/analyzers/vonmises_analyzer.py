from core.base_parser import BaseParser
from utilities.analyzer import get_atom_group_indices
import numpy as np

class VonMisesAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.group_indices = None

        self._average_stress_cache = None
        self._max_stress_cache = None
        self._min_stress_cache = None

    def get_stress_evolution(self):
        if self._average_stress_cache is not None:
            return self._average_stress_cache, self._max_stress_cache, self._min_stress_cache
        
        average_stress = []
        max_stress = []
        min_stress = []

        for data in self.parser.get_data():
            stress = data[:, 5]
            average_stress.append(np.mean(stress))
            max_stress.append(np.max(stress))
            min_stress.append(np.min(stress))

        self._average_stress_cache = np.array(average_stress)
        self._max_stress_cache = np.array(max_stress)
        self._min_stress_cache = np.array(min_stress)

        return self._average_stress_cache, self._max_stress_cache, self._min_stress_cache
    
    def get_stress_evolution_by_group(self, group='all'):
        '''
        Args:
            group: ('lower_plane', 'upper_plane', 'nanoparticle', 'all')
        '''
        group_indices = get_atom_group_indices(self.parser, -1)[group]
        average_stress = []
        max_stress = []
        min_stress = []
        
        for data in self.parser.get_data():
            stress = data[group_indices, 5]
            average_stress.append(np.mean(stress))
            max_stress.append(np.max(stress))
            min_stress.append(np.min(stress))
        
        return np.array(average_stress), np.array(max_stress), np.array(min_stress)