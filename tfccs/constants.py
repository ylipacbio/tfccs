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
MIN_NUMPASSES = 1
MAX_NUMPASSES = 2000

DEFAULT_VALIDATION_CSV = "/pbi/dept/secondary/siv/testdata/ccsqv/Mule/hg2/hg2_validation_5pct_err.fextract.csv"
HG2_GRC38_HIGHCONFIDENCE_NOINCONSISTENT = '/pbi/dept/consensus/ccsqv/data/Mule/hg2/hg2.Grch38.hc.bed'
HG2_GRC38_KNOWN_VARIANTS = '/pbi/dept/consensus/ccsqv/data/Mule/hg2/hg2.Grch38.variants.bed'
