import config
import os
import pandas as pd
import pickle
from loguru import logger

logger.add(os.path.join(config.Paths._LOGS, "pipeline.log"), rotation="10 MB", level="DEBUG", enqueue=True, backtrace=True, diagnose=True)
# logger.remove(0)  # Remove the default logger to prevent logs from being printed to the terminal


class FileOperator:
    def __init__(self):
        self.SAVE_PATH = config.Paths._DATA
        self._ensure_directories()
        self._check_sim_results()

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
        if filename.startswith("empty"):
            filepath = os.path.join(self.SAVE_PATH,config.SimulationNames.EMPTY,filename)
        else:
            filepath = os.path.join(self.SAVE_PATH,config.SimulationNames.LOADED,filename)
        return pd.read_csv(filepath, encoding="latin-1")
    
    def _ensure_directories(self):
        """
        Ensure that the directories in the given path exist.

        :param path: The path to check and create directories for.
        """
        attrs = [attr for attr in config.Paths().__dir__() if attr.startswith("_") and not attr.startswith('__')]
        dirs = [config.Paths().__getattribute__(attr) for attr in attrs]
        for directory in dirs:
            os.makedirs(directory,exist_ok=True)

    def _check_sim_results(self):
        total_loss = 0
        for path in [config.Paths._EMPTY,config.Paths._LOADED]:
            wagon_type = path.split("/")[-1]
            # print(wagon_type)
            for way_type in config.SimulationNames.WAY_TYPES:
                for profile in config.SimulationNames.PROFILES:
                    for fault in config.SimulationNames.FAULTS:
                        file_to_check = os.path.join(path,f"{wagon_type}_{way_type}_{profile}_{fault}.csv") if fault != "" else os.path.join(path,f"{wagon_type}_{way_type}_{profile}.csv") 
                        if not os.path.exists(file_to_check):
                            logger.debug(f"missing: {file_to_check}")
                            total_loss+=1
                        else:
                            continue
        if total_loss>0:
            logger.debug(f"total missing files: {total_loss}")
