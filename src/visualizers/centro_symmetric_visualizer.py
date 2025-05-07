from core.base_parser import BaseParser
from analyzers.centro_symmetric_analyzer import CentroSymmetricAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class CentroSymmetricVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = CentroSymmetricAnalyzer(parser)
        # Color map for centro-symmetric parameter
        self.centro_symmetric_cmap = 'viridis'
        # Colors for different structure classifications
        self.structure_colors = {
            'perfect': 'blue',
            'partial_defect': 'cyan',
            'stacking_fault': 'green',
            'surface': 'yellow',
            'defect': 'red'
        }