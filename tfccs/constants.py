# Define important re-usable varaibles

NO_TRAIN_FEATURES = [r.strip() for r in """
Movie
HoleNumber
CCSPos
CCSLength
ArrowQv
CCSToGenomeStrand
CCSToGenomeCigar
PrevCcsToGenomeCigar
NextCcsToGenomeCigar
CcsToGenomePrevDeletions
Insertion0_FWD
Insertion0_REV
""".split('\n') if len(r.strip())]


ORDERED_FEATURES_KEY = "OrderedFeatures"
BASE_FEATURE_STAT_KEY = "BaseFeatureStat"
BASE_MAP_PROBABILITY_KEY = "BaseMapProbability"

MIN_DIST2END = 100
ALLOWED_STRANDS = "F"
ALLOWED_CIGARS = "IX="
