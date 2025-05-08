import os


class Common:
    SEED: int = 101


class Paths:

    WORKDIR = os.path.dirname(os.getcwd())  # Dissertation # fmt:skip
    DATA = os.path.join(WORKDIR, "data")
    _EMPTY = os.path.join(DATA, "empty")
    _LOADED = os.path.join(DATA, "loaded")

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
