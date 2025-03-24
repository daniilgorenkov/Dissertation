import config
import os
import pandas as pd
import pickle


class FileOperator:
    def __init__(self):
        self.SAVE_PATH = config.Paths.DATA

    def save(self, obj, filename: str):
        """
        Save an object to a file using pickle.

        :param obj: The object to save.
        :param filename: The name of the file to save the object to.
        """
        filepath = os.path.join(self.SAVE_PATH, filename + ".pkl")
        with open(filepath, "wb") as file:
            pickle.dump(obj, file)

    def load(self, filename: str):
        """
        Load an object from a file using pickle.

        :param filename: The name of the file to load the object from.
        :return: The loaded object.
        """
        filepath = os.path.join(self.SAVE_PATH, filename + ".pkl")
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def load_csv(self, filename: str):
        """
        Load a CSV file into a pandas DataFrame.

        :param filename: The name of the CSV file to load.
        :return: A pandas DataFrame containing the CSV data.
        """
        filepath = os.path.join(self.SAVE_PATH, filename + ".csv")
        return pd.read_csv(filepath, encoding="latin-1")
