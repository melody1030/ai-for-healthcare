"""Fixed rules for IsoHarmonix.

The proposal says all rules (number of steps, distance, spread cutoff) are
fixed BEFORE the user study. They all live here so they are easy to find,
report, and freeze. Do not change these values after the pilot.
"""

# Mood scale used by both the DEAM ratings and the user's mood sliders.
SCALE_MIN = 1.0
SCALE_MAX = 9.0

# Number of songs in each playlist (iso-guided and direct-to-target).
N_STEPS = 5

# Distance used to match a target point to a song in valence-arousal space.
DISTANCE_METRIC = "euclidean"

# Rater-disagreement cutoff. Each song gets a spread score:
#     spread = (valence_std + arousal_std) / 2
# Songs with a spread ABOVE this quantile are dropped as possible mislabels.
# 0.90 means we drop the 10% of songs with the largest disagreement.
SPREAD_QUANTILE = 0.90

# A matched song further than this from its target point is flagged as a
# "coverage gap" in reports and in the why-card (it is still played).
GAP_WARNING = 1.0

# Default folder where the Kaggle DEAM archive is unzipped.
DEFAULT_DATA_DIR = "data/DEAM"

# Default file for the cleaned song table.
DEFAULT_CLEAN_CSV = "data/songs_clean.csv"
