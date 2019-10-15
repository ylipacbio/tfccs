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
