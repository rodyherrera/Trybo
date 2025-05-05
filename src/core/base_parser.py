from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np

class BaseParser(ABC):
    def __init__(self, filename: str):
        self.filename = filename
        self.data = None
        self.box_size = None
        self.column_names = []
        self.metadata = {}

    @abstractmethod
    def parse(self) -> Tuple[np.ndarray, List, List[str]]:
        '''Parse the file and return data, box size, and column names'''
        pass

    def get_data(self) -> np.ndarray:
        if self.data is None:
            self.data, self.box_size, self.column_names = self.parse()
        return self.box_size
    
    def get_box_size(self) -> List:
        if self.box_size is None:
            self.data, self.box_size, self.column_names = self.parse()
        return self.box_size
    
    def get_column_names(self) -> List[str]:
        if not self.column_names:
            self.data, self.box_size, self.column_names = self.parse()
        return self.column_names
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata