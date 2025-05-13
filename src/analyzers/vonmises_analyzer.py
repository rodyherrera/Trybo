from core.base_parser import BaseParser
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
        
        timesteps = self.parser.get_timesteps()
        average_stress = []
        max_stress = []
        min_stress = []

        for i in range(len(timesteps)):
            stress = self.parser.get_analysis_data('vonmises', i)
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
        timesteps = self.parser.get_timesteps()
        average_stress = []
        max_stress = []
        min_stress = []
        
        for i in range(len(timesteps)):
            # Obtener datos del grupo para este timestep
            data = self.parser.get_data(i)
            group_indices = self.parser.get_atom_group_indices(data)[group]
            
            # Usar get_analysis_data para obtener valores de estrés von Mises
            stress = self.parser.get_analysis_data('vonmises', i)
            # Aplicar filtro de grupo
            stress = stress[group_indices]
            
            average_stress.append(np.mean(stress))
            max_stress.append(np.max(stress))
            min_stress.append(np.min(stress))
        
        return np.array(average_stress), np.array(max_stress), np.array(min_stress)