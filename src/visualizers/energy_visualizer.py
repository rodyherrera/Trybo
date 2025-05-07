from core.base_parser import BaseParser
from analyzers.energy_visualizer import EnergyAnalyzer
import numpy as np

class EnergyVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = EnergyAnalyzer(parser)

        # Color maps for different energy types
        self.energy_cmaps = {
            'kinetic': 'plasma',
            'potential': 'viridis', 
            'total': 'turbo'
        }