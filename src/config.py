import os


class Paths:

    WORKDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))  # /mnt/c/Users/Daniil # fmt:skip
    DESKTOP = os.path.join(WORKDIR, "Desktop")
    SIMULATION_RESULTS = os.path.join(DESKTOP, "simulation_results")
    VERICAL_FORCE = os.path.join(SIMULATION_RESULTS, "vertical_force")
    SIDE_FORCE = os.path.join(SIMULATION_RESULTS, "side_force")
