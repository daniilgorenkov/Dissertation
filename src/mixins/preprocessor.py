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

    def _split(self, df: pd.DataFrame, idxs: list, vertical: bool = True):
        """Split DataFrame by indices into blocks"""
        results = []

        if len(idxs) <= 1:
            return results

        # Выбираем нужный столбец в зависимости от типа
        force_col = df.columns[1] if vertical else df.columns[2]  # или правильный индекс

        n_blocks = 12
        start_block = 0 if vertical else n_blocks
        end_block = n_blocks if vertical else 2 * n_blocks

        for i in range(start_block, end_block):
            if i + 1 >= len(idxs):
                continue

            start, end = idxs[i], idxs[i + 1]
            speed = round((i - start_block + 1) * 10 / 3.6, 2)

            block_df = df.iloc[start:end][["time_step", force_col]].copy()
            block_df = block_df.dropna(subset=[force_col])

            if block_df.empty:
                continue

            block_df.set_index("time_step", inplace=True)

            force_type = "Vertical" if vertical else "Side"
            col_name = f"{force_type} {speed}"
            block_df = block_df.rename(columns={force_col: col_name})
            results.append(block_df)

        return results

    def _split_data(self, df: pd.DataFrame, idxs: list) -> tuple[list, list]:

        verticals = self._split(df, idxs)
        sides = self._split(df, idxs, False)

        for v, s in zip(verticals, sides):
            print(v.columns[0], v.min().values[0], "|", s.columns[0], s.min().values[0])
        return verticals, sides

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

    def update_column_names(self, dfs, hook=False):
        for df in dfs:
            self.rename_duplicated_columns(df)

        if hook:
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
        v_s = []
        for s, v in zip(side_features, vertical_features):
            v = v.add_prefix("vertical_").reset_index(drop=True)
            s = s.add_prefix("side_").reset_index(drop=True)
            v_s.append(pd.concat([v, s], axis=1))
        return pd.concat(v_s, axis=0)

    def set_dtypes(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        for df in dfs:
            # Set each column's dtype from config.Preprocessor.DTYPES_OUT
            for col, dtype in config.Preprocessor.DTYPES_OUT.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
        return dfs

    def preprocess_file_results(self, filepath: str, hook: bool = False) -> pd.DataFrame:
        """
        Preprocess the results of a simulation file.

        :param filename: The name of the file to preprocess.
        :return: A DataFrame containing the preprocessed data.
        """
        logger.debug(f"Preprocessing file: {filepath}")
        self.is_straight = True if "straight" in os.path.basename(filepath).split("_") else False
        df = self.load_csv(filepath)  # load csv file
        df = self._reset_column_names(df)  # as column names are weird clean them
        idxs = self._get_split_index(
            df
        )  # as simulation results go one by one in one column, searching for indexes where split them
        df_vertical, df_side = self._split_data(df, idxs)  # split the data into smaller DataFrames based on the indices

        # Here we will split separated dfs into smaller dfs based on wheel rotation time
        df_vertical_all = self.split_by_time(df_vertical)
        df_side_all = self.split_by_time(df_side)

        # update column names
        df_vertical_all = self.update_column_names(df_vertical_all, hook)
        df_side_all = self.update_column_names(df_side_all, hook)

        # now as we have splitted dfs we need to extract features from them
        vertical_features = self.extract_features(df_vertical_all)
        side_features = self.extract_features(df_side_all)

        vertical_features = self.set_dtypes(vertical_features)
        side_features = self.set_dtypes(side_features)

        combined_forces = (
            self.combine_forces(vertical_features, side_features)
            .reset_index(drop=True)
            .replace(0.0, config.Preprocessor.ZERO_VALUE)
        )

        # Add target column based on filename
        filename = os.path.basename(filepath)
        fault_target = 1 if any(keyword in filename for keyword in config.SimulationNames.FAULTS) else 0

        filename_profile = [profile for profile in config.SimulationNames.PROFILES if profile in filename]

        if filename_profile:
            profile_target = config.SimulationNames.PROFILE_TARGET.get(filename_profile[0], 0)
        else:
            print(f"Profile not found in filename: {filename}")
        combined_forces["fault_target"] = fault_target
        combined_forces["profile_target"] = profile_target

        gc.collect()  # Force garbage collection to free up memory
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
