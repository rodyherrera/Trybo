from core.base_parser import BaseParser
import numpy as np

class EnergyAnaylizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self._atom_groups = None