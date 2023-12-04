masks = [
    (240.811, 246.635),
    (338.457, 340.797),
    (382.138, 387.859),
    (473.184, 492.427),
    (849, 905.574),
]

oxide_ranges = {
    "SiO2": {
        "Full": (0, 100),
        "Low": (0, 50),
        "Mid": (30, 70),
        "High": (60, 100),
    },
    "TiO2": {"Full": (0, 100), "Low": (0, 2), "Mid": (1, 5), "High": (3, 100)},
    "Al2O3": {
        "Full": (0, 100),
        "Low": (0, 12),
        "Mid": (10, 25),
        "High": (20, 100),
    },
    "FeOT": {
        "Full": (0, 100),
        "Low": (0, 15),
        "Mid": (5, 25),
        "High": (15, 100),
    },
    "MgO": {
        "Full": (0, 100),
        "Low": (0, 3.5),
        "Mid": (0, 20),
        "High": (8, 100),
    },
    "CaO": {"Full": (0, 42), "Low": (0, 7), "Mid": (0, 15), "High": (30, 100)},
    "Na2O": {"Full": (0, 100), "Low": (0, 4), "High": (3.5, 100)},
    "K2O": {"Full": (0, 100), "Low": (0, 2), "High": (1.5, 100)},
}


major_oxides = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeOT",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
]


paper_individual_sm_rmses = {
    "Full": {
        "SiO2": 5.66,
        "TiO2": 0.51,
        "Al2O3": 2.79,
        "FeOT": 3.34,
        "MgO": 1.43,
        "CaO": 1.8,
        "Na2O": 0.6,
        "K2O": 0.78,
    },
    "Low": {
        "SiO2": 4.24,
        "TiO2": 0.26,
        "Al2O3": 1.58,
        "FeOT": 2.08,
        "MgO": 0.47,
        "CaO": 1.17,
        "Na2O": 0.51,
        "K2O": 0.32,
    },
    "Mid": {
        "SiO2": 3.46,
        "TiO2": 0.61,
        "Al2O3": 1.89,
        "FeOT": 1.97,
        "MgO": 1.33,
        "CaO": 1.49,
        # Na2O and K2O are not included in the mid range
    },
    "High": {
        "SiO2": 4.07,
        "TiO2": 6.09,
        "Al2O3": 2.03,
        "FeOT": 3.68,
        "MgO": 2.43,
        "CaO": 2.38,
        "Na2O": 0.54,
        "K2O": 0.74,
    },
}
