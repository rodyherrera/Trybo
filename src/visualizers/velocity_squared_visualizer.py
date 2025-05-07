from core.base_parser import BaseParser
from analyzers.velocity_squared_analyzer import VelocitySquaredAnalyzer
import numpy as np
import seaborn as sns

class VelocitySquaredVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        self.analyzer = VelocitySquaredAnalyzer(parser)