from abc import ABC
from typing import List, Dict, Any, Union
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from numba import njit
from core.helpers import init_worker, _load_segment
import numpy as np
import gc
import mmap

class BaseParser(ABC):
    # For reduce footprint
    __slots__ = (
        'filename',
        '_mm',
        '_timesteps',
        '_timestep_atom_info',
        '_headers',
        '_atoms_spatial_coordinates_indices',
        '_analysis_column_map',
        '_metadata',
        '_box_bounds'
    )

    def __init__(self, filename: str):
        self.filename = filename
        file = open(self.filename, 'r+b')
        self._mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

        self._timesteps: List[int] = []
        self._timestep_atom_info: Dict[int, tuple[int, int]] = {}

        self._headers: List[str] = []
        self._atoms_spatial_coordinates_indices: List[int] = []
        self._analysis_column_map: Dict[str, Union[int, List[int]]] = {}

        self._metadata: Dict[str, Any] = {}
        self._box_bounds = None

        self._index_file()

    def _index_file(self):
        size = self._mm.size()
        offset = 0
        while True:
            position_timestep = self._mm.find(b'ITEM: TIMESTEP', offset)
            if position_timestep < 0:
                break
            self._mm.seek(position_timestep)
            self._mm.readline()
            timestep_line = self._mm.readline()
            try:
                timestep = int(timestep_line.strip())
            except ValueError:
                break
            self._timesteps.append(timestep)
            position_atoms = self._mm.find(b'ITEM: ATOMS', self._mm.tell())
            if position_atoms < 0:
                break
            self._mm.seek(position_atoms)
            # ITEM: ATOMS id type x y z ...
            atom_header = self._mm.readline()
            if not self._headers:
                parts = atom_header.split()[2:]
                self._headers = [p.decode() for p in parts]
                self._parse_atoms_spatial_coordinates_indices()
                self._create_analysis_column_map()
            data_start = self._mm.tell()
            next_item = self._mm.find(b'ITEM:', data_start)
            data_end = next_item if next_item >= 0 else size
            self._timestep_atom_info[timestep] = (data_start, data_end)
            offset = data_end

        if not self._headers:
            raise ValueError('No headers were found in the file. Check the format.')
        
    @lru_cache(maxsize=8)
    def _load_timestep_data(self, timestep: int) -> np.ndarray:
        if timestep not in self._timestep_atom_info:
            raise ValueError(f'Timestep {timestep} not found in file.')
        start, end = self._timestep_atom_info[timestep]
        segment = self._mm[start:end]
        flat = np.fromstring(segment, sep=' ')
        n_cols = len(self._headers)
        try:
            return flat.reshape(-1, n_cols)
        except ValueError:
            raise ValueError(f'Data reshape error for timestep {timestep}: {flat.size} values for {n_cols} columns.')

    def _parse_atoms_spatial_coordinates_indices(self):
        try:
            x_idx = self._headers.index('x')
            y_idx = self._headers.index('y')
            z_idx = self._headers.index('z')
        except ValueError:
            print('WARNING: The headers do not contain "x", "y", or "z". This is a critical error, and nothing may work as expected. Set the expected positions (2, 3, and 4, respectively).')
            x_idx = 2
            y_idx = 3
            z_idx = 4
        self._atoms_spatial_coordinates_indices = [x_idx, y_idx, z_idx]

    def _load_segment(args):
        filename, start, end, n_cols = args
        with open(filename, 'r+b') as file:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            segment = mm[start:end]
            mm.close()
        flat = np.fromstring(segment, sep=' ')
        return flat.reshape(-1, n_cols)

    def _load_all_timesteps(self):
        args = [
            (start, end, len(self._headers))
            for start, end in self._timestep_atom_info.values()
        ]
        with ProcessPoolExecutor(
            initializer=init_worker,
            initargs=(self.filename,)
        ) as exe:
            return list(exe.map(_load_segment, args))
        
    def _create_analysis_column_map(self):
        analysis_headers = {
            'centro_symmetric': 'c_center_symmetric',
            'cna': 'c_cna',
            'vonmises': 'v_atoms_stress',
            'velocity_squared': 'v_velocity_squared_atom',
            'cluster': 'c_cluster',
            'ke_hotspots': 'c_ke_hotspots',
            'is_hotspot': 'v_is_hotspot',
            'coord': 'c_coord',
            'ptm': ['c_ptm[1]', 'c_ptm[2]', 'c_ptm[3]', 'c_ptm[4]', 'c_ptm[5]', 'c_ptm[6]']
        }

        for analysis_type, header in analysis_headers.items():
            if isinstance(header, list):
                idxs = [self._headers.index(h) for h in header if h in self._headers]
                if idxs:
                    self._analysis_column_map[analysis_type] = idxs
            elif header in self._headers:
                self._analysis_column_map[analysis_type] = self._headers.index(header)
        print(f'Analysis column map created: {self._analysis_column_map}')
    
    def get_analysis_data(self, analysis_type: str, timestep_idx: int = -1) -> Union[np.ndarray, List[np.ndarray]]:
        if analysis_type not in self._analysis_column_map:
            raise ValueError(f'Analysis type "{analysis_type}" not found in headers: {self._headers}')
        
        column_idx = self._analysis_column_map[analysis_type]

        if timestep_idx < 0:
            timestep_idx = len(self._timesteps) + timestep_idx
        
        timestep_data = self._get_timestep_data(timestep_idx)

        if isinstance(column_idx, list):
            # Return multiple columns for array data
            return [timestep_data[:, i] for i in column_idx]
        
        return timestep_data[:, column_idx]
    
    def _get_timestep_data(self, timestep_idx):
        if timestep_idx < 0:
            timestep_idx = len(self._timesteps) + timestep_idx
        timestep = self._timesteps[timestep_idx]
        return self._load_timestep_data(timestep)

    def clear_data_cache(self):
        self._load_timestep_data.cache_clear()
        gc.collect()

    def get_atom_group_indices(self, data):
        '''
        Identifies atom indices for different groups based on their position along the Z axis
        
        Args:
            data: Data array to use instead of the parser's data

        Returns:
            dict: Dictionary with indices for each group (lower_plane, upper_plane, nanoparticle, all)
        '''
        _, _, z = self.get_atoms_spatial_coordinates(data)
        lower, upper = BaseParser._njit_group_indices(z)
        nanoparticle = ~(lower | upper)
        return {
            'lower_plane': np.where(lower)[0],
            'upper_plane': np.where(upper)[0],
            'nanoparticle': np.where(nanoparticle)[0],
            'all': np.arange(data.shape[0])
        }
        
    def get_atoms_spatial_coordinates(self, data):
        x_idx, y_idx, z_idx = self._atoms_spatial_coordinates_indices
        x = data[:, x_idx]
        y = data[:, y_idx]
        z = data[:, z_idx]
        return x, y, z
    
    def get_data(self, timestep_idx = None) -> np.ndarray:
        if timestep_idx is None:
            print('WARNING: Loading all timesteps into memory. This may consume a lot of RAM.')
            return self._load_all_timesteps()
        
        if isinstance(timestep_idx, int):
            return self._get_timestep_data(timestep_idx)
    
        if timestep_idx in self._timesteps:
            return self._load_timestep_data(timestep_idx)

        raise ValueError(f'Timestep {timestep_idx} not found')
    
    def get_headers(self) -> List[str]:
        return self._headers
    
    def get_timesteps(self) -> Dict[str, Any]:
        return self._timesteps
    
    def get_column_data(self, column_name: str, timestep_idx: int = -1) -> np.ndarray:
        if column_name not in self._headers:
            raise ValueError(f'Column {column_name} does not exists in headers.')
        column = self._headers.index(column_name)
        return self._get_timestep_data(timestep_idx)[:, column]

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    def get_atom_types(self, timestep_idx=-1) -> np.ndarray:
        column = self._headers.index('type')
        return self._get_timestep_data(timestep_idx)[:, col].astype(np.int64)

    def get_atom_count(self, timestep_idx: int = -1) -> int:
        return self._get_timestep_data(timestep_idx).shape[0]
    
    @staticmethod
    @njit
    def _njit_group_indices(z: np.ndarray):
        z_min = z.min()
        z_max = z.max()
        n = z.shape[0]
        lower = np.empty(n, np.bool_)
        upper = np.empty(n, np.bool_)
        for i in range(n):
            lower[i] = z[i] <= z_min + 2.5
            upper[i] = z[i] >= z_max - 2.5
        return lower, upper
    
    def __del__(self):
        try:
            self._mm.close()
        except Exception:
            pass