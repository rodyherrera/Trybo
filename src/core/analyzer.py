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