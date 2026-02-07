import config
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from mixins.file_operator import FileOperator
from mixins.utils import set_logger
from tqdm import tqdm
from scipy.fft import fft
from imblearn.over_sampling import SMOTENC
from mixins.utils import cats_first_floats_later, standardize_float_columns, is_float, apply_prefix_to_dtype_dict
import gc
import uuid


logger = set_logger(config.Paths._LOGS)


class Preprocessor(FileOperator):
    def __init__(self):
        super().__init__()
        self.functions = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]

    def _reset_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reset the column names of a DataFrame to be lowercase and snake case.

        :param df: The DataFrame to reset the column names of.
        """
        columns = df.columns
        self.new_cols = []
        for col in columns:
            for char in config.Preprocessor.BAD_CHARS:
                col: str = col.replace(char, "")
            if col not in config.Preprocessor.IGNORE_COLUMNS:
                # clear column name string and round speed value
                split_col = col.split(" ")
                clean_split = [sp for sp in split_col if sp != ""]
                assert is_float(clean_split[1]), f"Second string value isn't float {clean_split[1]}"
                rounded_speed = round(float(clean_split[1]), 2)
                col = f"{clean_split[0]} {rounded_speed}"
                self.new_cols.append(col)
        self.new_cols.insert(0, "time_step")
        df.columns = self.new_cols

        return df

    def _get_split_index(self, df: pd.DataFrame) -> list:
        """
        Get the index at which to split the DataFrame into the train and test sets.

        :param df: The DataFrame to split.
        :return: The index at which to split the DataFrame. If the DataFrame has 12 speeds, must return 24 indexes, if 8 speeds - 16 indexes.
                It depends on `-1` number so it must be static.

        Example
        -------
        >>> df = pd.DataFrame({'time_step': [0, 0.1, 0.2, 15, 15.1, 15.2],'Vertical 2.78':[0, 1, 2, 3, 4, 5]})
        >>> self._get_split_index(df)
        [0, 3, 6]
        """

        idxs = df[df["time_step"].diff() < -1].index.tolist()
        idxs.insert(0, 0)
        idxs.append(df.shape[0])

        return idxs

    def split_side_vertical(self, df, indexes, neg_threshold=0.4, min_magnitude_ratio=0.1):
        """
        Classifies segments as 'side' or 'vertical' using sign + magnitude behavior
        """

        values = df.iloc[:, 1].to_numpy()
        result = {"side": [], "vertical": []}

        global_scale = np.percentile(np.abs(values), 90)

        for i in range(len(indexes) - 1):
            start, end = indexes[i], indexes[i + 1]
            seg = values[start:end]

            if len(seg) == 0:
                continue

            neg_ratio = np.mean(seg < 0)
            mean_mag = np.mean(np.abs(seg))

            # rule that mimics visual intuition
            is_side = neg_ratio >= neg_threshold or mean_mag < min_magnitude_ratio * global_scale

            label = "side" if is_side else "vertical"
            result[label].append((start, end))

        return result

    def _get_rotation_indices_for_column(self, df: pd.DataFrame, col: str) -> np.ndarray:
        """
        Calculate split indices for a single column based on wheel rotation time.

        :param df: The DataFrame containing the data.
        :param col: The column name (e.g., "Vertical 5.56").
        :return: An array of index positions where the DataFrame should be split.
        """
        spl = col.split(" ")
        # print(f"spl: {spl}")
        cleaned_list = [item for item in spl if item.strip()]
        # print(f"cleaned list: {cleaned_list}")
        speed = float(cleaned_list[1])  # Will raise if the format is wrong

        length = 2 * np.pi * config.WagonParams.WHEEL_RADIUS
        t = length / speed
        max_index = df.index.max()
        n_slices = int(max_index // t)

        # print(f"Speed: {speed:.2f}, Wheel rotation time: {t:.4f}, Slices: {n_slices}")
        return np.linspace(0, max_index, n_slices + 1)

    def split_df_by_time_indices(self, df: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Split all columns in the DataFrame into segments based on wheel rotation time.

        :param df: The DataFrame with multiple speed-based columns.
        :return: List of segmented DataFrames.
        """
        all_segments = []

        cols = df.columns[0]
        indices = self._get_rotation_indices_for_column(df, cols)

        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            mask = (df.index >= start) & (df.index < end)
            segment = df.loc[mask, cols].copy()
            if not segment.empty:
                segment.name = f"{cols}_{i}"  # Make column name unique
                segment = segment.reset_index(drop=True)
                all_segments.append(segment)

        return pd.concat(all_segments, axis=1)

    def _split(self, df: pd.DataFrame, idxs: dict, vertical: bool = True):
        """
        Split DataFrame into force blocks by indices and direction.

        :param df: Input DataFrame with cleaned column names
        :param idxs: Dict with keys 'side' and 'vertical',
                    values are lists of (start, end) index tuples
        :param vertical: If True extract vertical forces, else side forces
        :return: List of DataFrames, one per force column
        """
        results = []

        # Select index ranges based on direction
        segment_key = "vertical" if vertical else "side"
        segments = idxs.get(segment_key, [])

        if not segments:
            return results

        # Get all non-time columns - ВСЕ колонки, не фильтруем по имени!
        # Потому что split_side_vertical уже определил ВРЕМЕННЫЕ сегменты
        all_cols = [col for col in df.columns if col != "time_step"]
        force_cols = all_cols  # Используем ВСЕ колонки

        # For each force column, extract data from relevant segments
        for col in force_cols:
            run_dfs = []

            for start_idx, end_idx in segments:
                block_df = df.iloc[start_idx:end_idx][["time_step", col]].copy()

                # Skip empty or NaN-only blocks
                if block_df[col].isna().all():
                    continue

                block_df = block_df.dropna(subset=[col])

                if not block_df.empty:
                    block_df.set_index("time_step", inplace=True)
                    run_dfs.append(block_df)

            # Concatenate all runs for this column
            if run_dfs:
                combined_df = pd.concat(run_dfs, axis=0)
                combined_df.index.name = "time_step"
                results.append(combined_df)

        return results

    def compute_statistical_features(self, df: pd.DataFrame) -> dict:
        stats = {}
        for col in df.columns:
            series = df[col]
            stats[col] = {
                "mean": series.mean(),
                "max": series.max(),
                "min": series.min(),
                "median": series.median(),
                "std": series.std(),
                "variance": series.var(),
                "skewness": series.skew(),
                "kurtosis": series.kurt(),
                "range": series.max() - series.min(),
                "percentile_25": series.quantile(0.25),
                "percentile_75": series.quantile(0.75),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
            }
        return stats

    def compute_temporal_features(self, df):

        temp_feats = {}
        for col in df.columns:
            raw_series = df[col]

            # Handle nested arrays or single-row vectors
            if isinstance(raw_series.iloc[0], (np.ndarray, list)):
                series = pd.Series(np.array(raw_series.iloc[0]).flatten())
            else:
                series = raw_series.dropna()

            values = series.values

            if values.ndim != 1:
                raise ValueError(f"Column '{col}' is not 1D. Got shape: {values.shape}")

            gradient = np.gradient(values)
            second_derivative = np.gradient(gradient)
            zero_crossings = np.where(np.diff(np.sign(gradient)))[0]
            peaks, _ = find_peaks(values)
            troughs, _ = find_peaks(-values)

            temp_feats[col] = {
                "first_derivative_mean": np.mean(gradient),
                "second_derivative_mean": np.mean(second_derivative),
                "num_zero_crossings": len(zero_crossings),
                "num_peaks": len(peaks),
                "num_troughs": len(troughs),
            }
        return temp_feats

    def compute_frequency_features(self, df: pd.DataFrame) -> dict:
        freq_feats = {}
        for col in df.columns:
            values = df[col].values.astype(float)
            values = values[~np.isnan(values)]  # Drop NaNs

            n = len(values)
            if n == 0:
                freq_feats[col] = {"dominant_frequency": 0, "spectral_energy": 0, "spectral_entropy": 0}
                continue

            fft_vals = np.abs(fft(values))[: n // 2]
            freqs = np.fft.fftfreq(n)[: n // 2]

            if np.sum(fft_vals) == 0:
                dominant_freq = 0
                spectral_energy = 0
                spectral_entropy = 0
            else:
                dominant_freq = freqs[np.argmax(fft_vals)]
                spectral_energy = np.sum(fft_vals**2)
                p = fft_vals / np.sum(fft_vals)
                spectral_entropy = -np.sum(p * np.log2(p + 1e-10))

            freq_feats[col] = {
                "dominant_frequency": dominant_freq,
                "spectral_energy": spectral_energy,
                "spectral_entropy": spectral_entropy,
            }

        return freq_feats

    def extract_features_from_force_df(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}

        stats = self.compute_statistical_features(df)
        temp = self.compute_temporal_features(df)
        freq = self.compute_frequency_features(df)

        for col in df.columns:
            features[col] = {**stats[col], **temp[col], **freq[col]}

        return pd.DataFrame(features).T  # return as a nice DataFrame

    def rename_duplicated_columns(self, df):
        counts = {}
        new_cols = []

        for col in df.columns:
            if col in counts:
                counts[col] += 1
                new_cols.append(f"{col}_{counts[col]}")
            else:
                counts[col] = 0
                new_cols.append(col)

        df.columns = new_cols

    def vertical_side(self, df: pd.DataFrame):
        """
        Making df with columns time_step, Vertical 2.78 Side 2.78

        :return:
        List of DataFrames where on index time_step and two columns: Vertical and Side forces
        """
        pair_dfs = []
        col_names = [col for col in df.columns if col != "time_step" and not col.startswith("Side")]
        for col in col_names:
            pair_cols = ["time_step", col, col.replace("Vertical", "Side")]
            valid_df: pd.DataFrame = df[pair_cols]
            valid_df.set_index("time_step", inplace=True)
            pair_dfs.append(valid_df)
        return pair_dfs

    def split_by_time(self, dfs: list[pd.DataFrame]) -> list:
        small_dfs = []
        for df in dfs:
            segments = self.split_df_by_time_indices(df)  # list of split DataFrames
            small_dfs.append(segments)
        return small_dfs
    def trim_segments(self, dfs: list[pd.DataFrame], head_trim: int = 0, tail_trim: int = 0) -> list[pd.DataFrame]:
        """
        Trim beginning and end of each segment to remove noise/transients.

        :param dfs: List of DataFrames to trim
        :param head_trim: Number of rows to remove from beginning
        :param tail_trim: Number of rows to remove from end
        :return: List of trimmed DataFrames
        """
        if head_trim <= 0 and tail_trim <= 0:
            return dfs
        
        trimmed = []
        for df in dfs:
            if df is None or df.empty:
                trimmed.append(df)
                continue
            
            start = head_trim if head_trim > 0 else 0
            end = -tail_trim if tail_trim > 0 else None
            df_trimmed = df.iloc[start:end].copy()
            trimmed.append(df_trimmed)
        
        return trimmed

    def update_column_names(self, dfs, hook=False):
        for df in dfs:
            self.rename_duplicated_columns(df)

        if hook and dfs:
            self.save(dfs, f"hook_{uuid.uuid4().hex[:3]}")

        return dfs

    def extract_features(self, dfs: list[pd.DataFrame]):
        features = []
        for part in dfs:
            # Extract features from each segment excpects that in df in columns are names "Vertical 8.3345..."
            feat_df = self.extract_features_from_force_df(part)  # shape: (num_columns, num_features)
            features.append(feat_df)
        return features

    def combine_forces(self, vertical_features: list[pd.DataFrame], side_features: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine vertical and side force features.

        If one list is empty, just return the other with appropriate prefixes.
        Otherwise, zip and concatenate pairs.
        """
        v_s = []

        # Handle case where one list is empty
        if not vertical_features and not side_features:
            return pd.DataFrame()  # Both empty

        if not side_features:
            # Only vertical features
            for v in vertical_features:
                v = v.add_prefix("vertical_").reset_index(drop=True)
                v_s.append(v)
        elif not vertical_features:
            # Only side features
            for s in side_features:
                s = s.add_prefix("side_").reset_index(drop=True)
                v_s.append(s)
        else:
            # Both have data - pair them up
            for s, v in zip(side_features, vertical_features):
                v = v.add_prefix("vertical_").reset_index(drop=True)
                s = s.add_prefix("side_").reset_index(drop=True)
                v_s.append(pd.concat([v, s], axis=1))

        return pd.concat(v_s, axis=0) if v_s else pd.DataFrame()

    def set_dtypes(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        for df in dfs:
            # Set each column's dtype from config.Preprocessor.DTYPES_OUT
            for col, dtype in config.Preprocessor.DTYPES_OUT.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
        return dfs

    def preprocess_file_results(self, filepath: str, hook: bool = False, head_trim: int = 500, tail_trim: int = 10) -> pd.DataFrame:
        """
        Preprocess the results of a simulation file.

        Pipeline:
        1. Load CSV and clean column names
        2. Find temporal boundaries (where simulation runs are separated)
        3. Split data by force direction (vertical/side), extracting data from all runs
        4. Split each force block by wheel rotation time
        5. Trim segments to remove noise/transients
        6. Extract statistical, temporal, and frequency features
        7. Combine and add target labels

        :param filepath: Path to the CSV file to preprocess.
        :param hook: If True, save intermediate results for debugging.
        :param head_trim: Number of rows to remove from beginning of each segment (default 500).
        :param tail_trim: Number of rows to remove from end of each segment (default 10).
        :return: A DataFrame with extracted features and target labels.
        """
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

        vertical_features = self.extract_features(vertical_segments)
        side_features = self.extract_features(side_segments)

        # ===== STEP 7: Set dtypes and combine =====
        vertical_features = self.set_dtypes(vertical_features)
        side_features = self.set_dtypes(side_features)

        combined_forces = (
            self.combine_forces(vertical_features, side_features)
            .reset_index(drop=True)
            .replace(0.0, config.Preprocessor.ZERO_VALUE)
        )

        # ===== STEP 8: Add target labels =====
        fault_target = 1 if any(kw in filename for kw in config.SimulationNames.FAULTS) else 0

        matching_profiles = [p for p in config.SimulationNames.PROFILES if p in filename]
        profile_target = config.SimulationNames.PROFILE_TARGET.get(matching_profiles[0], 0) if matching_profiles else 0

        combined_forces["fault_target"] = fault_target
        combined_forces["profile_target"] = profile_target

        gc.collect()
        return combined_forces

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

        dfs = []
        for fpath in tqdm(all_paths, desc="Preprocessing files", total=n_files):
            hook = False
            if fpath in loaded_paths and "straight" in fpath and "polzun" not in fpath and "ellips" not in fpath:
                hook = True
            df = self.preprocess_file_results(fpath, hook)
            dfs.append(df)

        # Concatenate all DataFrames into one
        final_df = pd.concat(dfs, axis=0)
        self.save(final_df, "preprocessed_data")

    def data_augmentation(self) -> pd.DataFrame:

        df: pd.DataFrame = self.load("preprocessed_data").fillna(config.Preprocessor.ZERO_VALUE)
        # Create a copy of the DataFrame for augmentation
        # As we have 2 types of targets we will make 2 augmentations for each target due to targets distributions
        CATEGORICAL_DTYPES: dict = apply_prefix_to_dtype_dict(
            config.Preprocessor.CATEGORICAL_DTYPES, config.Preprocessor.PREFIXES
        )
        NUMERICAL_DTYPES: dict = apply_prefix_to_dtype_dict(
            config.Preprocessor.NUMERICAL_DTYPES, config.Preprocessor.PREFIXES
        )

        for target in tqdm(config.Preprocessor.TARGETS, desc="Augmenting data", total=len(config.Preprocessor.TARGETS)):
            X = df.drop(config.Preprocessor.TARGETS, axis=1).copy()
            feature_cols = X.columns
            y = df[target].copy()

            categorical_columns = [X.columns.get_loc(col) for col in list(CATEGORICAL_DTYPES.keys())]

            smotenc = SMOTENC(
                categorical_features=categorical_columns,
                sampling_strategy=config.Preprocessor.SAMPLE_STRATEGY,
                random_state=config.Common.SEED,
            )
            # Apply SMOTENC to the DataFrame
            X_aug, y_aug = smotenc.fit_resample(X, y)

            # Create augmented DataFrame
            augmented_df = pd.DataFrame(X_aug, columns=feature_cols)
            augmented_df[target] = y_aug

            for col in [col for col in CATEGORICAL_DTYPES.keys() if col not in config.Preprocessor.TARGETS]:
                if col in augmented_df.columns:
                    augmented_df[col] = augmented_df[col].round().clip(lower=config.Preprocessor.ZERO_VALUE)

            augmented_df = augmented_df.astype(
                {
                    **CATEGORICAL_DTYPES,
                    **NUMERICAL_DTYPES,
                }
            )

            # Standardize float columns
            augmented_floats = standardize_float_columns(
                augmented_df, ignore_cols=list(CATEGORICAL_DTYPES.keys()) + [target]
            )
            augmented_df[augmented_floats.columns] = augmented_floats
            # Reorder columns to have categorical columns first
            augmented_df = cats_first_floats_later(augmented_df)

            # Store and save augmented DataFrame
            self.save(augmented_df, f"preprocessed_data_{target}")
            gc.collect()  # Force garbage collection to free up memory

    def preprocess(self):
        """
        Preprocess the data by loading, cleaning, and saving it.
        """

        # if self.is_data_preprocessed() == False:
        #     print(self.is_data_preprocessed())
        self.preprocess_all_files()
        # self.data_augmentation()
        logger.debug("Data is already preprocessed, " "Start training models")
