from core.base_parser import BaseParser
import matplotlib.pyplot as plt
import numpy as np

class VonmisesVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
