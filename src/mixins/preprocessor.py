import config
import os
import pandas as pd
from mixins.file_operator import FileOperator
from tqdm import tqdm 
from scipy.fft import fft
import numpy as np
from scipy.signal import find_peaks
from loguru import logger

logger.add(os.path.join(config.Paths._LOGS, "pipeline.log"), rotation="10 MB", level="DEBUG", enqueue=True, backtrace=True, diagnose=True)
logger.remove(0)  # Remove the default logger to prevent logs from being printed to the terminal

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
                self.new_cols.append(col)
        self.new_cols.insert(0, "time_step")
        df.columns = self.new_cols
        return df

    def _get_split_index(self, df: pd.DataFrame) -> list:
        """
        Get the index at which to split the DataFrame into the train and test sets.

        :param df: The DataFrame to split.
        :return: The index at which to split the DataFrame.
        """
        idxs = df[df["time_step"].diff() < -10].index.tolist()
        idxs.insert(0, 0)
        idxs.append(df.shape[0])
        return idxs
    
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
        logger.debug(f"total simulation results: {len(sim_results)}")
        return sim_results
    
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
    
    def _set_index(self, dfs: list[pd.DataFrame]) -> list:
        """
        Set the index of the DataFrames in the given list to the time_step column.

        :param dfs: The list of DataFrames to set the index of.
        :return: The list of DataFrames with the index set.
        """
        for df in dfs:
            df.set_index("time_step", inplace=True)
 
    
    def _wheel_rotation_time(self, df: pd.DataFrame) -> np.ndarray:
        speed = float(df.columns[0].split(" ")[1])
        lenght = 2 * np.pi * config.WagonParams.WHEEL_RADIUS
        t = lenght / speed
        max_indx = df.index.max()
        n_slices = int(max_indx//t)
        indexes = np.linspace(0, max_indx, n_slices)
 
        return indexes
    
    def split_df_by_time_indices(self,df:pd.DataFrame) -> pd.DataFrame:
        indices = self._wheel_rotation_time(df)
        
        segments = []
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            segment = df.loc[start:end]
            segments.append(segment)
        
        for i, seg in enumerate(segments):
            n = len(seg)
            # calcular paso promedio (asumiendo time_step constante)
            step = seg.index.to_series().diff().median()
            new_index = np.arange(0, n * step, step)[:n]
            seg.index = new_index
            segments[i] = seg

        if len(segments) <= 1:
            logger.debug(f"Only one or less segment found!")
        return pd.concat(segments,axis=1)
    
    def _rename_columns(self, dfs: list[pd.DataFrame]) -> list:
        """
        Rename the columns of the DataFrames in the given list.

        :param dfs: The list of DataFrames to rename the columns of.
        :return: The list of DataFrames with renamed columns.
        """
        for df, col in zip(dfs, self.new_cols[1:]):
            df.columns = ["time_step", col]
 
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


    def preprocess_file_results(self, filename: str) -> pd.DataFrame:
        """
        Preprocess the results of a simulation file.

        :param filename: The name of the file to preprocess.
        :return: A DataFrame containing the preprocessed data.
        """
        df = self.load_csv(filename)
        df = self._reset_column_names(df)
        idxs = self._get_split_index(df)
        dfs = self._split_data(df, idxs)
        self._rename_columns(dfs)
        self._set_index(dfs)
        logger.debug(f"start extract features")
        for i in range(len(dfs)):
            dfs[i] = self.split_df_by_time_indices(dfs[i])
            dfs[i] = self.rename_duplicated_columns(dfs[i])
            dfs[i] = self.extract_features_from_force_df(dfs[i])
        return dfs
    
    def preprocess_all(self,filenames: list[str]) -> dict:
        """"
        
        Preprocess all simulation files in the given list.
        Args:
            filenames (list[str]): A list of file names to preprocess.
            Returns:
            DataFrame: A DataFrame containing the preprocessed data from all files."""
        n_files = len(filenames)
        dfs = []
        for filename in tqdm(filenames, desc="Preprocessing files", total=n_files):
                df = self.preprocess_file_results(filename)
                dfs.append(df)
        return pd.concat(dfs, axis=0)


    
