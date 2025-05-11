import os
from site import PREFIXES
import numpy as np


class Common:
    SEED: int = 101


class Paths:

    WORKDIR = os.path.dirname(os.getcwd())  # Dissertation # fmt:skip
    DATA = os.path.join(WORKDIR, "data")
    _EMPTY = os.path.join(DATA, "empty")
    _LOADED = os.path.join(DATA, "loaded")
    _LOGS = os.path.join(WORKDIR, "logs")


class WagonParams:
    WHEEL_RADIUS = 0.475  # m

class SimulationNames:
    LOADED = "loaded"
    EMPTY = "empty"
    POLZUN = "polzun"
    ELLIPS = "ellips"
    FAULTS = [POLZUN, ELLIPS]
    PROFILES = ["greb30","greb28","greb26","greb24","newwagonwh","gost"]
    PROFILE_TARGET = {"newwagonwh":0,
                      "gost":0,
                      "greb30":1,
                      "greb28":1,
                      "greb26":2,
                      "greb24":2}


class SimulationSpeeds:
    STRAIGHT = [i for i in range(10, 130, 10)]
    CURVE = [i for i in range(10, 90, 10)]


class Preprocessor:
    BAD_CHARS = [
        "XÂðåìÿ (ñåê) (Âðåìÿ  ñåê)Q(V)_1l ",
        "- [Ýêñïåðèìåíò: ",
        "]  (Q(V)_1l)",
        "Âðåìÿ (ñåê) (Âðåìÿ  ñåê)Q(V)_1l ",
        "v0=",
        "X ",
        "]"

    ]
    IGNORE_COLUMNS = ["Unnamed: 24", "Unnamed: 16"]

    ZERO_VALUE = 1e-6
    
    PREFIXES = ["vertical","side"]

    NUMERICAL_DTYPES = {
    'mean': np.float32,
    'max': np.float32,
    'min': np.float32,
    'median': np.float32,
    'std': np.float32,
    'variance': np.float32,
    'skewness': np.float32,
    'kurtosis': np.float32,
    'range': np.float32,
    'percentile_25': np.float32,
    'percentile_75': np.float32,
    'iqr': np.float32,
    'first_derivative_mean': np.float32,
    'second_derivative_mean': np.float32,
    'dominant_frequency': np.float32,
    'spectral_energy': np.float32,
    'spectral_entropy': np.float32,
    }

    CATEGORICAL_DTYPES = {
    'num_zero_crossings': np.int32,
    'num_peaks': np.int32,
    'num_troughs': np.int32,
    }
    
    TARGET_DTYPES = {
    'fault_target': np.int32,
    'profile_target': np.int32
    }

    DTYPES_OUT:dict = {
    **NUMERICAL_DTYPES,
    **CATEGORICAL_DTYPES,
    **TARGET_DTYPES
    }


    TARGETS = TARGET_DTYPES.keys()
    CATEGORICAL_COLS = CATEGORICAL_DTYPES.keys()
    NUMERICAL_COLS = NUMERICAL_DTYPES.keys()
    
    SAMPLE_STRATEGY = "auto"
