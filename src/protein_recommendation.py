# utils/protein_recommendation.py

import numpy as np

# Species-specific protein percentages in feed
PROTEIN_TABLE = {
    'Bream': 0.38,
    'Roach': 0.33,
    'Pike': 0.45,
    'Smelt': 0.40,
    'Perch': 0.38,
    'Parkki': 0.37,
    'Whitefish': 0.33
}

# Species-specific optimal temperature ranges (°C)
TEMP_TABLE = {
    'Bream': (18, 22),
    'Roach': (17, 20),
    'Pike': (16, 20),
    'Smelt': (14, 18),
    'Perch': (15, 20),
    'Parkki': (17, 21),
    'Whitefish': (10, 15)
}

# Species constants for expected weight (from your health classifier)
SPECIES_CONSTS = {
    "Bream": (0.0094, 3.2545),
    "Roach": (0.012, 3.1),
    "Perch": (0.01, 3.2),
    "Pike": (0.007, 3.3),
    "Smelt": (0.008, 3.1),
    "Whitefish": (0.009, 3.25),
    "Parkki": (0.010, 3.15)
}

def temperature_factor(species, current_temp):
    """Returns a temperature adjustment factor (0 < factor <= 1)"""
    min_temp, max_temp = TEMP_TABLE[species]
    if min_temp <= current_temp <= max_temp:
        return 1.0
    if current_temp < min_temp:
        factor = max(0.5, 1 - (min_temp - current_temp) * 0.05)
    else:
        factor = max(0.5, 1 - (current_temp - max_temp) * 0.05)
    return factor

def recommend_feed(species, current_weight, length3, health_status,
                   fcr=1.5, temp=20, growth_days=10):
    """
    Calculate daily feed and protein to reach healthy target weight.
    length3: fish's third length measurement
    health_status: output from health classifier ("Malnourished"/"Healthy"/"Overweight")
    """
    if species not in PROTEIN_TABLE or species not in SPECIES_CONSTS:
        raise ValueError(f"Species '{species}' not recognized")

    a, b = SPECIES_CONSTS[species]
    expected_weight = a * (length3 ** b)

    # Determine target weight
    if health_status == "Malnourished":
        target_weight = expected_weight
    else:
        target_weight = current_weight  # no feed required if healthy/overweight

    if current_weight >= target_weight:
        return f"{species} is healthy. No additional feed required."

    # Weight gap
    weight_gap = target_weight - current_weight

    # Daily feed
    daily_feed = (weight_gap * fcr) / growth_days

    # Adjust for temperature
    temp_factor = temperature_factor(species, temp)

    # Daily protein requirement
    protein_percent = PROTEIN_TABLE[species]
    daily_protein = daily_feed * protein_percent * temp_factor

    # Temperature advisory
    min_temp, max_temp = TEMP_TABLE[species]
    if temp < min_temp:
        temp_advisory = f"Warning: Temperature below optimal range ({min_temp}-{max_temp}°C). Growth may slow."
    elif temp > max_temp:
        temp_advisory = f"Warning: Temperature above optimal range ({min_temp}-{max_temp}°C). Growth may slow."
    else:
        temp_advisory = "Temperature within optimal range."

    return {
        "Species": species,
        "Current Weight (g)": round(current_weight, 2),
        "Target Weight (g)": round(target_weight, 2),
        "Weight Gap (g)": round(weight_gap, 2),
        "Daily Feed (g/day)": round(daily_feed, 2),
        "Daily Protein (g/day)": round(daily_protein, 2),
        "Growth Duration (days)": growth_days,
        "Temperature Advisory": temp_advisory
    }
