from typing import Dict, Any, Optional
from core.base_parser import BaseParser
import numpy as np

class StatisticalAnalyzer:
    def __init__(self, parser: BaseParser, property_column: Optional[int] = None):
        self.parser = parser
        self.property_column = property_column
        
    def analyze(self, threshold: float = None) -> Dict[str, Any]:
        data = self.parser.get_data()
        
        # Determine property column if not specified
        if self.property_column is None:
            # Try to get property column from specialized parser
            if hasattr(self.parser, 'get_property_column_index'):
                self.property_column = self.parser.get_property_column_index()
            else:
                # Default to last column (common in LAMMPS dumps)
                self.property_column = data.shape[1] - 1
        
        # Extract property values
        values = data[:, self.property_column]
        
        # Calculate statistics
        stats = {
            'mean': np.mean(values),
            'median': np.median(values),
            'max': np.max(values),
            'min': np.min(values),
            'std': np.std(values)
        }
        
        # Add threshold-based statistics if threshold provided
        if threshold is not None:
            high_count = np.sum(values > threshold)
            high_percentage = (high_count / len(values)) * 100
            stats.update({
                'threshold': threshold,
                'high_count': high_count,
                'high_percentage': high_percentage
            })
        
        return stats