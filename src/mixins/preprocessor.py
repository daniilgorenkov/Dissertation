import config
import os
import pandas as pd
import numpy as np
from mixins.file_operator import FileOperator


class Preprocessor(FileOperator):
    def __init__(self):
        super().__init__()

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

        print(f"total simulation results: {len(sim_results)}")
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
        return pd.concat(dfs, axis=1)
