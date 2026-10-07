"""Simple rule-of-thumb labels derived from a temperature.

These are fixed temperature thresholds, not a trained model and not a
measurement of water chemistry. They are shown in the app as rough guidance.
"""

from sst_data import to_lon360

# Specific seas first, big oceans last, so small regions are not swallowed.
_REGIONS = [
    ("Arctic Ocean",       65,  90, -180, 180),
    ("Southern Ocean",    -90, -50, -180, 180),
    ("Arabian Sea",         5,  25,   55,  75),
    ("Bay of Bengal",       5,  25,   80, 100),
    ("South China Sea",     0,  25,  100, 125),
    ("Mediterranean Sea",  30,  45,   -5,  40),
    ("Caribbean Sea",      10,  25,  -85, -60),
    ("Indian Ocean",      -50,  30,   20, 120),
    ("Atlantic Ocean",    -50,  65,  -70,  20),
    ("Pacific Ocean",     -50,  65,  120, 180),
    ("Pacific Ocean",     -50,  65, -180, -70),
]


def get_region(lat, lon):
    lon = to_lon360(lon)
    if lon > 180:
        lon -= 360
    for name, lat_min, lat_max, lon_min, lon_max in _REGIONS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "Open Ocean"


def sst_category(sst):
    if sst < 10:
        return "Cold"
    if sst < 20:
        return "Moderate"
    if sst < 28:
        return "Warm"
    if sst < 32:
        return "High"
    return "Extreme"


def coral_health(sst):
    if sst > 30:
        return "Bleaching risk"
    if sst > 28:
        return "Heat stress"
    return "No heat stress"


def water_quality_index(sst):
    if 20 <= sst <= 28:
        return 90, "Excellent"
    if 15 <= sst <= 32:
        return 70, "Good"
    if 10 <= sst <= 35:
        return 50, "Moderate"
    return 30, "Poor"


SPECIES = [
    ("Coral reef", 20, 28), ("Tuna", 5, 30), ("Shrimp", 10, 30),
    ("Sea turtle", 20, 32), ("Seagrass", 15, 30), ("Reef fish", 22, 28),
    ("Shark", 8, 26), ("Whale", 2, 20), ("Penguin", -2, 15),
    ("Squid", 5, 25), ("Crab", 5, 25),
]


def species_table(sst):
    return [
        {"Species": name, "Comfortable range": f"{lo} to {hi} °C",
         "Status": "Suitable" if lo <= sst <= hi else "Outside range"}
        for name, lo, hi in SPECIES
    ]
