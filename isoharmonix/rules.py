"""Study rules fixed before the pilot.

The proposal says these are frozen and then tested on new participants.
Change them only with team agreement. The notebook's "Fixed rules" cell
imports this module so the values live in one place.
"""

# Same 1-9 scale for DEAM static ratings and the mood sliders.
SCALE_MIN = 1.0
SCALE_MAX = 9.0

# Songs per playlist. Iso-guided and direct-to-target both use this length.
N_STEPS = 5

# Distance in the valence-arousal plane. The retriever implements this metric.
DISTANCE_METRIC = "euclidean"

# Drop songs above this quantile of rater spread (the top 10%).
SPREAD_QUANTILE = 0.90

# A chosen song further than this from its waypoint is flagged on the why-card.
GAP_WARNING = 1.0

ISO = "iso"
DIRECT = "direct"

# joint: order-independent minimum-cost assignment across both playlists.
# iso_first: legacy rule, iso playlist takes songs before direct.
# direct_first: the reversed legacy rule, used only to measure order bias.
ALLOCATION_JOINT = "joint"
ALLOCATION_ISO_FIRST = "iso_first"
ALLOCATION_DIRECT_FIRST = "direct_first"
ALLOCATIONS = (
    ALLOCATION_JOINT,
    ALLOCATION_ISO_FIRST,
    ALLOCATION_DIRECT_FIRST,
)
