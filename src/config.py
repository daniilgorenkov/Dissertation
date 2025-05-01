import os


class Common:
    SEED: int = 101


class Paths:

    WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Dissertation # fmt:skip
    _DATA = os.path.join(WORKDIR, "data")
    _EMPTY = os.path.join(_DATA, "empty")
    _LOADED = os.path.join(_DATA, "loaded")
    _LOGS = os.path.join(WORKDIR, "logs")

class WagonParams:
    WHEEL_RADIUS = 0.475  # m

class SimulationNames:
    FORCE_VERTICAL = "vertical"
    FORCE_SIDE = "side"
    LOADED = "loaded"
    EMPTY = "empty"
    GREB_24 = "greb24"
    GREB_26 = "greb26"
    GREB_30 = "greb30"
    GREB_GOST = "gost"
    GREB_UM = "newwagonwh"
    STRAIGHT = "straight"
    CURVE_350 = "curve_350"
    CURVE_650 = "curve_650"
    POLZUN = "polzun"
    ELLIPS = "ellips"
    WAY_TYPES = [STRAIGHT,CURVE_350,CURVE_650]
    FAULTS = [POLZUN,ELLIPS,""]
    PROFILES = [GREB_24,GREB_26,GREB_30,GREB_GOST,GREB_UM]



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
        "X_"
    ]
    IGNORE_COLUMNS = ["Unnamed: 24", "Unnamed: 16"]
    ZERO = 1e-10  # To avoid division by zero

    DTYPES_OUT = {
        'mean': 'float32', 'max': 'float32', 'min': 'float32', 'median': 'float32', 
        'std': 'float32', 'variance': 'float32', 'skewness': 'float32', 
        'kurtosis': 'float32', 'range': 'float32', 'percentile_25': 'float32', 
        'percentile_75': 'float32', 'iqr': 'float32', 'first_derivative_mean': 'float32', 
        'second_derivative_mean': 'float32', 'num_zero_crossings': 'int32', 
        'num_peaks': 'int32', 'num_troughs': 'int32', 'dominant_frequency': 'float32', 
        'spectral_energy': 'float32', 'spectral_entropy': 'float32', 'target': 'int32'
    }


class Trainer:
    TRAINING_SIZE = 0.8
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    EPOCHS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001