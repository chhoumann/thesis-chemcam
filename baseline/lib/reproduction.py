masks = [
    (240.811, 246.635),
    (338.457, 340.797),
    (382.138, 387.859),
    (473.184, 492.427),
    (849, 905.574),
]

spectrometer_wavelength_ranges = {
    "UV": (223.4, 325.97),
    "VIO": (381.86, 471.03),
    "VNIR": (494.93, 927.06),
}

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

training_info = {
    "SiO2": {
        "Full": {"range": (0, 100), "n_components": 6, "normalization": 1},
        "Low": {"range": (0, 50), "n_components": 9, "normalization": 3},
        "Mid": {"range": (30, 70), "n_components": 6, "normalization": 3},
        "High": {"range": (60, 100), "n_components": 5, "normalization": 1},
    },
    "TiO2": {
        "Full": {"range": (0, 100), "n_components": 5, "normalization": 3},
        "Low": {"range": (0, 2), "n_components": 7, "normalization": 3},
        "Mid": {"range": (1, 5), "n_components": 5, "normalization": 1},
        "High": {"range": (3, 100), "n_components": 3, "normalization": 3},
    },
    "Al2O3": {
        "Full": {"range": (0, 100), "n_components": 6, "normalization": 3},
        "Low": {"range": (0, 12), "n_components": 6, "normalization": 1},
        "Mid": {"range": (10, 25), "n_components": 8, "normalization": 1},
        "High": {"range": (20, 100), "n_components": 6, "normalization": 1},
    },
    "FeOT": {
        "Full": {"range": (0, 100), "n_components": 8, "normalization": 3},
        "Low": {"range": (0, 15), "n_components": 3, "normalization": 3},
        "Mid": {"range": (5, 25), "n_components": 8, "normalization": 1},
        "High": {"range": (15, 100), "n_components": 3, "normalization": 3},
    },
    "MgO": {
        "Full": {"range": (0, 100), "n_components": 7, "normalization": 3},
        "Low": {"range": (0, 3.5), "n_components": 6, "normalization": 1},
        "Mid": {"range": (0, 20), "n_components": 9, "normalization": 1},
        "High": {"range": (8, 100), "n_components": 8, "normalization": 1},
    },
    "CaO": {
        "Full": {"range": (0, 42), "n_components": 8, "normalization": 3},
        "Low": {"range": (0, 7), "n_components": 9, "normalization": 1},
        "Mid": {"range": (0, 15), "n_components": 9, "normalization": 3},
        "High": {"range": (30, 100), "n_components": 6, "normalization": 3},
    },
    "Na2O": {
        "Full": {"range": (0, 100), "n_components": 8, "normalization": 1},
        "Low": {"range": (0, 4), "n_components": 7, "normalization": 1},
        "High": {"range": (3.5, 100), "n_components": 7, "normalization": 1},
    },
    "K2O": {
        "Full": {"range": (0, 100), "n_components": 4, "normalization": 3},
        "Low": {"range": (0, 2), "n_components": 6, "normalization": 3},
        "High": {"range": (1.5, 100), "n_components": 9, "normalization": 1},
    },
}


optimized_blending_ranges = {
    "SiO2": {
        "Low": (-20.0, 17.48),
        "Low-Mid": (17.48, 55.63),
        "Mid": (55.63, 55.71),
        "Mid-High": (55.71, 70.3),
        "High": (70.3, 120.0),
    },
    "TiO2": {
        "Low": (-20.0, 0.85),
        "Low-Mid": (0.85, 1.9),
        "Mid": (1.9, 4.67),
        "Mid-High": (4.67, 7.29),
        "High": (7.29, 120.0),
    },
    "Al2O3": {
        "Low": (-20.0, 4.6),
        "Low-Mid": (4.6, 13.7),
        "Mid": (13.7, 18),
        "Mid-High": (18, 25.4),
        "High": (25.4, 120),
    },
    "FeOT": {
        "Low": (-20.0, 3.4),
        "Low-Mid": (3.4, 16.5),
        "Mid": (16.5, 16.5),
        "Mid-High": (16.5, 28.6),
        "High": (28.6, 120),
    },
    "MgO": {
        "Low": (-20, 0.46),
        "Low-Mid": (0.46, 6.2),
        "Mid": (6.2, 6.3),
        "Mid-High": (6.3, 17.8),
        "High": (17.8, 120),
    },
    "CaO": {
        "Low": (-20, 3.2),
        "Low-Mid": (3.2, 5.9),
        "Mid": (5.9, 16.2),
        "Mid-High": (16.2, 33),
        "High": (33, 120),
    },
    "Na2O": {
        "Low": (-20, 1.8),
        "Mid-High": (1.8, 4.9),
        "High": (4.9, 120),
    },
    "K2O": {
        "Low": (-20, 1.3),
        "Mid-High": (1.3, 3.0),
        "High": (3.0, 120),
    },
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

# Weights for blending PLS1-SM and ICA predictions
weighted_sum_oxide_percentages = {
    "Al2O3": {"PLS1-SM": 75, "ICA": 25},
    "FeOT": {"PLS1-SM": 75, "ICA": 25},
    # SiO2: Initial mix, changes to 75/25 based on conditions
    "SiO2": {"PLS1-SM": 50, "ICA": 50},
    "Na2O": {"PLS1-SM": 40, "ICA": 60},
    "K2O": {"PLS1-SM": 25, "ICA": 75},
    # DON'T KNOW - NOT MENTIONED
    "TiO2": {"PLS1-SM": 50, "ICA": 50},
    "MgO": {"PLS1-SM": 50, "ICA": 50},
    "CaO": {"PLS1-SM": 50, "ICA": 50},
}


# Manual mapping of sample names to composition names for samples that don't match folder names
# Sample (dir) name -> composition name
folder_to_composition_sample_name = {
    "vs211681": "VS-2116-81",
    "sancs2": "SANC-S",  # guess due to text in csv describing it, and it's the only unused SAN-C
    "sh5": "SH-5",
    # "7tio2": "",
    "sh59": "SH-59",
    # "75tio2": "",
    "ncsdc28041": "NCS-DC28041",
    "sanck": "SanC-K",
    # "3tio2": "",
    "sanca": "SanC-A",
    "sh73": "SH-73",
    # "50tio2": "",
    # "10tio2": "",
    "ncsdc47008": "NCS-DC47008",
    "nau1": "NAU-1",
    "sancb": "SanC-B",
    "kga2meds": "KGa-2-med-S",
    "nat18": "NAT-18",
    "idbdf": "ID_BDF",
    # "5tio2": "",
    "sancc": "SanC-C",
    "sancj": "SanC-J",
    # "0.1tio2": "",
    "nau2los": "NAu-2-low-s",
    "icel009010": "icel009-010",
    "nau2meds": "NAu-2-mid-s",
    "ncsdc47009": "NCS-DC47009",
    "sanci": "SanC-I",
    # "25tio2": "",
    # "1tio2": "",
    "bir1a": "BIR-1a",
}
