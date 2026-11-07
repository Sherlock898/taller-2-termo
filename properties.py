import numpy as np

COMPONENTS = {
    "IPA": {
        "antoine": {'A': 12.0717, 'B': 3640.2, 'C': -53.54},
        "Tc": 508.3,    # K
        "Pc": 47.91,    # bar
        "Vc": 2.22e-4,   # m3/mol
        "tsonopoulos": {'a': 0.0878, 'b': 0.0537, 'w': 0.665}
    },
    "Water": {
        "antoine": {"A": 11.9647, "B": 3984.92, "C": -39.724},
        "Tc": 647.13,      # K
        "Pc": 219.4,       # bar
        "Vc": 5.60e-5,     # m3/mol
        "tsonopoulos": {"a": 0.0279, "b": 0.0229, "w": 0.344}
    },
    "Acetone": {
        "antoine": {"A": 9.5543, "B": 2673.3, "C": -49.15},
        "Tc": 508.20,      # K
        "Pc": 47.09,       # bar
        "Vc": 2.09e-4,     # m3/mol
        "tsonopoulos": {"a": -0.03090, "b": 0.0000, "w": 0.304}
    }
}

NRTL_PAIRS = {
    ("IPA", "Water"): {
        "Aij": -1.3115,
        "Aji": 6.8284,
        "Bij": 426.3978,
        "Bji": -1483.4573,
        "alpha": 0.3
    },
    ("IPA", "Acetone"): {
        "Aij": 2.4494,
        "Aji": -2.4106,
        "Bij": -583.3452,
        "Bji": 822.4892,
        "alpha": 0.3
    },
    ("Water", "Acetone"): {
        "Aij": 0.0544,
        "Aji": 6.3981,
        "Bij": 419.9716,
        "Bji": -1808.991,
        "alpha": 0.3
    }
}

VIRIAL_PAIRS = {
    ("IPA", "Water"): {"kij": 0.0},
    ("IPA", "Acetone"): {"kij": 0.0},
    ("Water", "Acetone"): {"kij": 0.0}
}

def get_mixture(components):
    n = len(components)

    if n == 0 or len(set(components)) != n:
        raise ValueError("Provide unique components")

    for comp in components:
        if comp not in COMPONENTS:
            raise ValueError(f"Unknown component: {comp}")

    # Antoine 
    ant = np.array([
        list(COMPONENTS[c]["antoine"].values())
        for c in components
    ], dtype=float)

    def fn_psats(T):
        return np.exp(ant[:, 0] - ant[:, 1] / (T + ant[:, 2]))

    # NRTL, virial interactions
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    alpha = np.zeros((n, n))
    kij = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = components[i], components[j]

            # NRTL 
            if (ci, cj) in NRTL_PAIRS:
                p = NRTL_PAIRS[(ci, cj)]
                A[i, j], A[j, i] = p["Aij"], p["Aji"]
                B[i, j], B[j, i] = p["Bij"], p["Bji"]
            elif (cj, ci) in NRTL_PAIRS:
                p = NRTL_PAIRS[(cj, ci)]
                A[i, j], A[j, i] = p["Aji"], p["Aij"]
                B[i, j], B[j, i] = p["Bji"], p["Bij"]
            else:
                raise ValueError(f"Missing NRTL parameters: {ci}, {cj}")
            alpha[i, j] = alpha[j, i] = p["alpha"]

            # Virial interactionss
            pair = (ci, cj) if (ci, cj) in VIRIAL_PAIRS else (cj, ci)
            if pair not in VIRIAL_PAIRS:
                raise ValueError(f"Missing virial parameters: {ci}, {cj}")
            kij[i, j] = kij[j, i] = VIRIAL_PAIRS[pair]["kij"]

    # Tsonopoulos
    def get_prop(key):
        return np.array([COMPONENTS[c][key] for c in components], dtype=float)

    def get_tson(key):
        return np.array([COMPONENTS[c]["tsonopoulos"][key]for c in components], dtype=float)

    nrtl_params = (A, B, alpha)

    virial_params = (get_tson("a"), get_tson("b"), get_tson("w"), kij, get_prop("Tc"), get_prop("Pc"), get_prop("Vc"))

    return fn_psats, nrtl_params, virial_params
