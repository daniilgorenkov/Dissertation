from fileinput import filename
import config
import os
import pandas as pd
import numpy as np
from mixins.file_operator import FileOperator
from tqdm import tqdm 
from scipy.signal import find_peaks
from scipy.fft import fft

class Preprocessor(FileOperator):
    def __init__(self):
        super().__init__()
        self.functions = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]
        # self.pbar = tqdm(total=len(self.functions), desc="Preprocessing")

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
        cleaned_list = [item for item in spl if item.strip()]
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

        for col in df.columns:
            # print(df.columns)
            indices = self._get_rotation_indices_for_column(df, col)

            for i in range(len(indices) - 1):
                start, end = indices[i], indices[i + 1]
                mask = (df.index >= start) & (df.index < end)
                segment = df.loc[mask, [col]].copy()
                if not segment.empty:
                    step = segment.index.to_series().diff().median()
                    new_index = np.arange(0, len(segment) * step, step)[:len(segment)]
                    segment.index = new_index
                    all_segments.append(segment)

        return pd.concat(all_segments,axis=1)



    def _split_data(self, df: pd.DataFrame, idxs: list):
        """
        Splits the given DataFrame into smaller DataFrames based on the provided indices.
        Args:
            df (pd.DataFrame): The DataFrame to be split.
            idxs (list): A list of indices defining the start and end points for each split.
        Returns:
            list: A list of DataFrames containing the split data.
        """
        sim_results = []

        for idx in range(len(idxs) - 1):
            start = idxs[idx]
            end = idxs[idx + 1]
            sim_results.append(df.iloc[start:end].iloc[:, :2])

        # print(f"total simulation results: {len(sim_results)}")
        
        return sim_results

    def _rename_columns(self, dfs: list[pd.DataFrame]) -> list:
        """
        Rename the columns of the DataFrames in the given list.

        :param dfs: The list of DataFrames to rename the columns of.
        :return: The list of DataFrames with renamed columns.
        """
        for df, col in zip(dfs, self.new_cols[1:]):
            df.columns = ["time_step", col]
        
        

    def _set_index(self, dfs: list[pd.DataFrame]) -> list:
        """
        Set the index of the DataFrames in the given list to the time_step column.

        :param dfs: The list of DataFrames to set the index of.
        :return: The list of DataFrames with the index set.
        """
        for df in dfs:
            df.set_index("time_step", inplace=True)
        

    def compute_statistical_features(self,df:pd.DataFrame) -> dict:
        stats = {}
        for col in df.columns:
            series = df[col]
            stats[col] = {
                'mean': series.mean(),
                'max': series.max(),
                'min': series.min(),
                'median': series.median(),
                'std': series.std(),
                'variance': series.var(),
                'skewness': series.skew(),
                'kurtosis': series.kurt(),
                'range': series.max() - series.min(),
                'percentile_25': series.quantile(0.25),
                'percentile_75': series.quantile(0.75),
                'iqr': series.quantile(0.75) - series.quantile(0.25)
            }
        return stats

    def compute_temporal_features(self,df):
        import numpy as np
        from scipy.signal import find_peaks

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
                'first_derivative_mean': np.mean(gradient),
                'second_derivative_mean': np.mean(second_derivative),
                'num_zero_crossings': len(zero_crossings),
                'num_peaks': len(peaks),
                'num_troughs': len(troughs)
            }
        return temp_feats
    
    def compute_frequency_features(self, df: pd.DataFrame) -> dict:
        freq_feats = {}
        for col in df.columns:
            values = df[col].values.astype(float)
            values = values[~np.isnan(values)]  # Drop NaNs

            n = len(values)
            if n == 0:
                freq_feats[col] = {
                    'dominant_frequency': 0,
                    'spectral_energy': 0,
                    'spectral_entropy': 0
                }
                continue

            fft_vals = np.abs(fft(values))[:n // 2]
            freqs = np.fft.fftfreq(n)[:n // 2]

            if np.sum(fft_vals) == 0:
                dominant_freq = 0
                spectral_energy = 0
                spectral_entropy = 0
            else:
                dominant_freq = freqs[np.argmax(fft_vals)]
                spectral_energy = np.sum(fft_vals ** 2)
                p = fft_vals / np.sum(fft_vals)
                spectral_entropy = -np.sum(p * np.log2(p + 1e-10))

            freq_feats[col] = {
                'dominant_frequency': dominant_freq,
                'spectral_energy': spectral_energy,
                'spectral_entropy': spectral_entropy
            }

        return freq_feats


    def extract_features_from_force_df(self,df:pd.DataFrame) -> pd.DataFrame:
        features = {}

        stats = self.compute_statistical_features(df)
        temp = self.compute_temporal_features(df)
        freq = self.compute_frequency_features(df)

        for col in df.columns:
            features[col] = {
                **stats[col],
                **temp[col],
                **freq[col]
            }

        return pd.DataFrame(features).T  # return as a nice DataFrame
    
    def rename_duplicated_columns(self,df:pd.DataFrame) -> pd.DataFrame:
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
        return df


    def preprocess_file_results(self, filepath: str) -> pd.DataFrame:
        """
        Preprocess the results of a simulation file.

        :param filename: The name of the file to preprocess.
        :return: A DataFrame containing the preprocessed data.
        """
        df = self.load_csv(filepath) # load csv file
        df = self._reset_column_names(df) # as column names are weird clean them
        idxs = self._get_split_index(df) # as simulation results go one by one in one column, searching for indexes where split them
        dfs = self._split_data(df, idxs) # split the data into smaller DataFrames based on the indices
        self._rename_columns(dfs) # rename columns to be more readable
        self._set_index(dfs) # set index name as time_step

        # at this point dfs looks like a list with dataframes where df are separated by forces and speeds
        # on index is time_step column named like "Vertical 8.3345..."
        # Here we will split separated dfs into smaller dfs based on wheel rotation time
        splitted_dfs = []
        for df in dfs:
            segments = self.split_df_by_time_indices(df)  # list of split DataFrames
            splitted_dfs.append(segments)
        
        # now as we have splitted dfs we need to extract features from them
        features = []
        for part in splitted_dfs:
            # mark duplicated columns with _1, _2, _3... etc.
            part = self.rename_duplicated_columns(part)
            # Extract features from each segment excpects that in df in columns are names "Vertical 8.3345..."
            feat_df = self.extract_features_from_force_df(part)  # shape: (num_columns, num_features)
            
            # Add target column based on filename
            filename = os.path.basename(filepath)
            fault_target = 1 if any(keyword in filename for keyword in config.SimulationNames.FAULTS) else 0
            
            filename_profile = [profile for profile in config.SimulationNames.PROFILES if profile in filename]
            profile_target = config.SimulationNames.PROFILE_TARGET.get(filename_profile[0])
            feat_df["fault_target"] = fault_target
            feat_df["profile_target"] = profile_target
            
            features.append(feat_df)

        return pd.concat(features, axis=0)

    
    
