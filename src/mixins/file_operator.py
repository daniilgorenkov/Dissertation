import config
import os
import pandas as pd
import pickle
from mixins.utils import set_logger
logger = set_logger(config.Paths._LOGS)


class FileOperator:
    def __init__(self):
        self.SAVE_PATH = config.Paths.DATA

    def is_data_preprocessed(self):
        file_names = os.listdir(config.Paths.DATA)
        files_check = []
        for names in ["preprocessed_data_profile_target.pkl","preprocessed_data_fault_target.pkl","preprocessed_data.pkl"]:
            if names in file_names:
                files_check.append(1)
        
        if sum(files_check) == 3:
            return True
        return False

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

    def load_csv(self, fpath: str):
        """
        Load a CSV file into a pandas DataFrame.

        :param filename: The name of the CSV file to load.
        :return: A pandas DataFrame containing the CSV data.

        Example:

        ```python
        from file_operator import FileOperator

        file_operator = FileOperator()
        path = '/home/daniil_gorenkov/dissertation/Dissertation/data/empty/empty_straight_greb30_ellips.csv'
        df = file_operator.load_csv(path)

        output:
        X Vertical - [Ýêñïåðèìåíò: v0=2.78]       Vertical - [Ýêñïåðèìåíò: v0=5.55727272727273]   \ 
0                                 0.000                           0.0 
1                                 0.005                           0.0
        ```

        """
        # filepath = os.path.join(self.SAVE_PATH, fpath + ".csv")
        return pd.read_csv(fpath, encoding="latin-1")
