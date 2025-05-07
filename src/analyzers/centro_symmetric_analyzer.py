from core.base_parser import BaseParser
import numpy as np

class CentroSymmetricAnalyzer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None
        # Defect thresholds for FCC (face-centered cubic) copper
        # Below this is considered perfect crystal
        self.perfect_threshold = 0.5
        # Above this is considered a significant defect
        self.defect_threshold = 8.0
        # Typical range for stacking faults in FCC
        self.stacking_fault_range = (2.0, 5.0)
        # Classification thresholds
        self.structure_ranges = {
            'perfect': (0, 0.5),
            'partial_defect': (0.5, 3.0),
            'stacking_fault': (3.0, 5.0),
            'surface': (5.0, 8.0),
            'defect': (8.0, float('inf'))
        }