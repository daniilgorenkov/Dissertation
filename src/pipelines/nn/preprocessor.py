from mixins.utils import set_logger
import config
import os
import numpy as np
import pandas as pd
from mixins.preprocessor import Preprocessor
from tqdm import tqdm
import re


logger = set_logger(config.Paths._LOGS)


class NNPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__()

    def get_vectors(self, filepath: str, hook: bool = False, head_trim: int = 500, tail_trim: int = 10):

        logger.debug(f"Preprocessing file: {filepath}")
        filename = os.path.basename(filepath)
        self.is_straight = "straight" in filename.split("_")

        # ===== STEP 1: Load and clean data =====
        df = self.load_csv(filepath)
        df = self._reset_column_names(df)

        # ===== STEP 2: Find temporal boundaries =====
        idxs = self._get_split_index(df)
        index_groups = self.split_side_vertical(df, idxs)

        # ===== STEP 3: Split by force direction =====
        vertical_blocks = self._split(df, index_groups, vertical=True)
        side_blocks = self._split(df, index_groups, vertical=False)

        # ===== STEP 4: Split by wheel rotation time =====
        vertical_segments = self.split_by_time(vertical_blocks)
        side_segments = self.split_by_time(side_blocks)

        # ===== STEP 5: Trim segments to remove noise =====
        vertical_segments = self.trim_segments(vertical_segments, head_trim=head_trim, tail_trim=tail_trim)
        side_segments = self.trim_segments(side_segments, head_trim=head_trim, tail_trim=tail_trim)

        # ===== STEP 6: Update column names and extract features =====
        vertical_segments = self.update_column_names(vertical_segments, hook)
        side_segments = self.update_column_names(side_segments, hook)

        return vertical_segments, side_segments

    def make_two_channel_turn(
        self, vertical_signal: np.ndarray, side_signal: np.ndarray, n_points: int = 1536
    ) -> np.ndarray:
        """
        vertical_signal: массив формы (M,)
        side_signal: массив формы (K,)
        n_points: сколько точек хотим после ресемплинга

        return: np.ndarray формы (2, n_points)
        """

        def resample(x, n):
            x = np.asarray(x, dtype=np.float32)
            if len(x) == 0:
                return np.zeros(n, dtype=np.float32)
            if len(x) == n:
                return x
            t_old = np.linspace(0, 1, len(x))
            t_new = np.linspace(0, 1, n)
            return np.interp(t_new, t_old, x).astype(np.float32)

        v = resample(vertical_signal, n_points)
        s = resample(side_signal, n_points)

        x = np.stack([v, s], axis=0)  # (2, N)
        return x

    def _suffix_key(self, col: str) -> str:
        """
        Достаём ключ оборота из имени колонки.
        Примеры:
        'Vertical 2.78_3' -> '3'
        'Side 2.78_3'     -> '3'
        """
        m = re.search(r"_(\d+)\s*$", col)
        return m.group(1) if m else col  # fallback

    def segments_to_tensor(self, vertical_segments, side_segments, n_points: int = 1536):
        X = []

        for v_df, s_df in zip(vertical_segments, side_segments):

            v_map = {self._suffix_key(c): c for c in v_df.columns}
            s_map = {self._suffix_key(c): c for c in s_df.columns}

            keys = sorted(set(v_map.keys()) & set(s_map.keys()), key=lambda k: int(k) if str(k).isdigit() else str(k))

            for k in keys:
                v = v_df[v_map[k]].dropna().values
                s = s_df[s_map[k]].dropna().values
                X.append(self.make_two_channel_turn(v, s, n_points))

        if not X:
            return np.zeros((0, 2, n_points), dtype=np.float32)

        return np.stack(X, axis=0)

    def make_labels_for_file(self, filename: str, n_samples: int):
        # как у тебя было
        fault_target = 1 if any(kw in filename for kw in config.SimulationNames.FAULTS) else 0

        matching_profiles = [p for p in config.SimulationNames.PROFILES if p in filename]
        profile_target = config.SimulationNames.PROFILE_TARGET.get(matching_profiles[0], 0) if matching_profiles else 0

        # продублировать на каждый оборот/пример
        y_fault = np.full((n_samples,), fault_target, dtype=np.int64)
        y_profile = np.full((n_samples,), profile_target, dtype=np.int64)
        return y_fault, y_profile

    def preprocess_all_files(self) -> pd.DataFrame:

        versions = os.listdir(config.Paths._EMPTY)

        empty_fnames = [os.listdir(os.path.join(config.Paths._EMPTY, v)) for v in versions]
        loaded_fnames = [os.listdir(os.path.join(config.Paths._LOADED, v)) for v in versions]

        # Create full paths
        empty_paths = [
            os.path.join(config.Paths._EMPTY, version, fname)
            for version, fnames in zip(versions, empty_fnames)
            for fname in fnames
        ]
        loaded_paths = [
            os.path.join(config.Paths._LOADED, version, fname)
            for version, fnames in zip(versions, loaded_fnames)
            for fname in fnames
        ]

        # Combine all paths into on list
        all_paths = empty_paths + loaded_paths
        n_files = len(all_paths)

        ds = []
        for fpath in tqdm(all_paths, desc="Preprocessing files", total=n_files):
            hook = False
            # if fpath in loaded_paths and "straight" in fpath and "polzun" not in fpath and "ellips" not in fpath:
            #     hook = True
            v_segs, s_segs = self.get_vectors(fpath, hook)
            X = self.segments_to_tensor(v_segs, s_segs, n_points=1536)
            y_fault, y_profile = self.make_labels_for_file(os.path.basename(fpath), X.shape[0])
            ds.append(
                {
                    "X": X,
                    "y_fault": y_fault,
                    "y_profile": y_profile,
                }
            )

        self.save(ds, "preprocessed_nn_data")

        return ds
