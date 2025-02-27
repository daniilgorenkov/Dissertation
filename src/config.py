import os


class Common:
    SEED: int = 101


class Paths:

    WORKDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))  # /mnt/c/Users/Daniil/Documents # fmt:skip
    DESKTOP = os.path.join(WORKDIR, "Desktop")
    SIMULATION_RESULTS = os.path.join(DESKTOP, "simulation_results")
    VERICAL_FORCE = os.path.join(SIMULATION_RESULTS, "vertical_force")
    SIDE_FORCE = os.path.join(SIMULATION_RESULTS, "side_force")


class SimulationNames:
    FORCE_VERTICAL = "vertical"
    FORCE_SIDE = "side"
    LOADED = "loaded"
    EMPTY = "empty"
    GREB_24 = "greb_24"
    GREB_26 = "greb_26"
    GREB_30 = "greb_30"
    GREB_GOST = "gost"
    GREB_UM = "newwagonw"
    STRAIGHT = "straight"
    CURVE_350 = "curve_350"
    CURVE_650 = "curve_650"
    POLZUN = "polzun15"
    ELLIPS = "ellips10"
    NO_FAULT = "normal"

class SimulationSpeeds:
    STRAIGHT = [i for i in range(10, 130, 10)]
    CURVE = [i for i in range(10, 90, 10)]