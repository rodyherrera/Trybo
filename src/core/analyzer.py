from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import os
import sys
import time
import yaml
import visualizers
import logging
import parsers

class Analyzer:
    def __init__(self, config_path: str = None, dump_folder: str = None):
        self.logger = logging.getLogger('Analyzer')
        self._setup_logging()

        # Initialize configuration
        self.config = {}
        if config_path:
            self.load_config(config_path)
        
        # Override dump folder if provided
        if dump_folder:
            self.set_dump_folder(dump_folder)
        
        # self.analysis_registry
    
    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    